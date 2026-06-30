"""
Cascade: M1 seg -> M2 morph -> M3 smooth (opt) -> M4 color/feature (boxes) -> M5 restore (opt).
  python face_beauty_pipeline_Z.py --show_gui
  python face_beauty_pipeline_Z.py --input img.png --boxes_json boxes.json [--use_member3]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from segmentation import segment_v4
from color_feature_enhance_Z import (
    ColorDetectParamsZ,
    EnhanceParamsZ,
    RegionBox,
    enhance_beauty_Z,
)

OUTPUT_DIR = "./output/output_member5"

try:
    from morphology_4_update import (
        MorphologyRefiner,
        PimpleProcessor,
        extract_eye_regions,
        extract_nose_regions,
    )
except ImportError:
    pass

try:
    from skin_smooth_freq_enhance import process_face_skin_smooth_and_freq_enhance

    MEMBER3_AVAILABLE = True
except ImportError:
    MEMBER3_AVAILABLE = False
    print("Warning: Member 3 module not found. Disabling member3.")


# --- Member 5 restoration -----------------------------------------------------

def _map_channels(img: np.ndarray, fn) -> np.ndarray:
    if img.ndim == 2:
        out = fn(img)
        return out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)
    chans = [fn(img[:, :, c]) for c in range(3)]
    out = np.stack(chans, axis=-1)
    return out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)


def member5_restoration(rgb: np.ndarray, method: str = "mean", kernel_size: int = 3) -> np.ndarray:
    method, k = method.lower(), int(kernel_size)
    if method == "mean":
        return _map_channels(rgb, lambda x: ndimage.uniform_filter(x.astype(np.float32), size=k))
    if method == "median":
        return _map_channels(rgb, lambda x: ndimage.median_filter(x, size=k))
    if method == "gaussian":
        return _map_channels(rgb, lambda x: ndimage.gaussian_filter(x.astype(np.float32), sigma=k / 2))
    if method == "vst":
        sig = max(0.5, k / 2.0)
        vst = 2.0 * np.sqrt(np.maximum(rgb.astype(np.float32), 0) + 3.0 / 8.0)
        if vst.ndim == 3:
            filt = np.stack(
                [ndimage.gaussian_filter(vst[:, :, c], sigma=sig) for c in range(3)], axis=-1
            )
        else:
            filt = ndimage.gaussian_filter(vst, sigma=sig)
        return np.clip((filt / 2.0) ** 2 - 3.0 / 8.0, 0, 255).astype(np.uint8)
    print(f"Unknown restoration method '{method}', using mean")
    return member5_restoration(rgb, "mean", k)


# --- Box JSON -----------------------------------------------------------------

def load_boxes_json(path: str) -> List[RegionBox]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        RegionBox(int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"]), b["kind"])  # type: ignore[arg-type]
        for b in data["boxes"]
    ]


def save_boxes_json(path: str, image_path: str, boxes: Sequence[RegionBox]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "image": image_path,
        "boxes": [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "kind": b.kind} for b in boxes],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_boxes_json_path(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(image_path)), f"{base}_boxes.json")


def resolve_member4_boxes(
    image_path: Optional[str],
    basename: str,
    *,
    explicit: Optional[Sequence[RegionBox]] = None,
    boxes_json: Optional[str] = None,
    boxes_dir: Optional[str] = None,
) -> Optional[List[RegionBox]]:
    if explicit:
        return list(explicit)
    paths: List[str] = []
    if boxes_json:
        paths.append(boxes_json)
    if image_path:
        b = os.path.splitext(os.path.basename(image_path))[0]
        d = os.path.dirname(os.path.abspath(image_path))
        paths.extend([os.path.join(d, f"{b}_boxes.json"), os.path.join(d, f"{b}.boxes.json")])
    if boxes_dir:
        paths.extend([os.path.join(boxes_dir, f"{basename}_boxes.json"), os.path.join(boxes_dir, f"{basename}.json")])
    for p in paths:
        if p and os.path.isfile(p):
            return load_boxes_json(p)
    return None


# --- Pipeline -----------------------------------------------------------------

@dataclass
class PipelineConfig:
    use_member2_pimple_removal: bool = False
    use_member3: bool = False
    member3_avg_radius: int = 2
    member3_gauss_sigma: float = 1.2
    member3_fft_smooth_sigma_norm: float = 0.07
    member3_spatial_mix: Tuple[float, float, float] = (0.35, 0.35, 0.30)
    member3_feather_radius: int = 14
    member3_skin_strength: float = 0.92
    use_member4: bool = True
    member4_whitening_strength: float = 0.85
    member4_enable_skin_tone: bool = True
    member4_enable_eyes_lips: bool = True
    member4_boxes: Optional[List[RegionBox]] = None
    member4_boxes_json: Optional[str] = None
    member4_boxes_dir: Optional[str] = None
    member4_color_params: Optional[ColorDetectParamsZ] = None
    member4_enhance_params: Optional[EnhanceParamsZ] = None
    use_member5_restoration: bool = False
    member5_restoration_method: str = "mean"
    member5_restoration_kernel_size: int = 3
    output_dir: str = OUTPUT_DIR
    save_intermediate: bool = False


class CascadeBeautyPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        self.morph_refiner = MorphologyRefiner(kernel_size=3)
        self.pimple_processor = PimpleProcessor() if config.use_member2_pimple_removal else None

    def _save(self, img: np.ndarray, name: str, suffix: str) -> None:
        if self.config.save_intermediate:
            path = os.path.join(self.config.output_dir, f"{name}_{suffix}.png")
            Image.fromarray(img).save(path)
            print(f"    Saved intermediate: {path}")

    def _member4_enhance_params(self) -> EnhanceParamsZ:
        base = self.config.member4_enhance_params or EnhanceParamsZ()
        wref = 0.85
        if (
            self.config.member4_enable_skin_tone
            and self.config.member4_enable_eyes_lips
            and abs(float(self.config.member4_whitening_strength) - wref) < 1e-6
            and self.config.member4_enhance_params is None
        ):
            return base
        fac = max(0.0, min(3.0, float(self.config.member4_whitening_strength))) / wref
        if not self.config.member4_enable_eyes_lips:
            base = replace(
                base, eye_strength=0.0, eye_v_lift=0.0, eye_contrast=0.0,
                lip_strength=0.0, lip_sat=1.0, lip_v_lift=0.0, lip_hue_shift=0.0,
            )
        if not self.config.member4_enable_skin_tone:
            base = replace(base, skin_strength=0.0, skin_whiten=1.0, skin_v_gain=1.0, skin_rgb_lift=0.0)
        else:
            base = replace(
                base,
                skin_strength=base.skin_strength * fac,
                skin_whiten=1.0 + (base.skin_whiten - 1.0) * fac,
                skin_v_gain=1.0 + (base.skin_v_gain - 1.0) * fac,
                skin_rgb_lift=base.skin_rgb_lift * fac,
            )
        return base

    def run(
        self,
        rgb: np.ndarray,
        basename: str = "image",
        *,
        image_path: Optional[str] = None,
        member4_boxes: Optional[Sequence[RegionBox]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        info: Dict[str, Any] = {"original_shape": rgb.shape}
        current = rgb.copy()

        print("  [Member1] Generating base skin mask...")
        base_mask_u8 = ((segment_v4(current)[0] > 127).astype(np.uint8) * 255)
        self._save(base_mask_u8, basename, "member1_mask")
        info["member1_mask"] = base_mask_u8

        print("  [Member2] Morphology refinement...")
        skin_bin = base_mask_u8 > 127
        eye_regions = extract_eye_regions(skin_bin)
        mask_no_pimples = base_mask_u8
        if self.pimple_processor is not None:
            print("  [Member2] Pimple removal...")
            pimple_mask = self.pimple_processor.detect_pimple_mask(
                base_mask_u8, eye_regions, extract_nose_regions(skin_bin), rgb=current
            )
            if np.any(pimple_mask):
                current = self.pimple_processor.apply_median_removal(current, pimple_mask)
                mask_no_pimples = self.pimple_processor.fill_pimple_holes_in_mask(base_mask_u8, pimple_mask)
        refined_mask = self.morph_refiner.refine_face_region(mask_no_pimples, eye_regions)
        info["skin_mask_final"] = refined_mask
        self._save(refined_mask, basename, "member2_skin_mask")

        if self.config.use_member3:
            if not MEMBER3_AVAILABLE:
                print("  [Member3] Skipped (module not available).")
            else:
                print("  [Member3] Frequency skin smoothing...")
                try:
                    current, aux3 = process_face_skin_smooth_and_freq_enhance(
                        current, skin_mask_u8=refined_mask,
                        avg_radius=self.config.member3_avg_radius,
                        gauss_sigma=self.config.member3_gauss_sigma,
                        fft_smooth_sigma_norm=self.config.member3_fft_smooth_sigma_norm,
                        spatial_mix=self.config.member3_spatial_mix,
                        feather_radius=self.config.member3_feather_radius,
                        skin_strength=self.config.member3_skin_strength,
                        enable_high_acne_removal=True,
                    )
                    info["member3_aux"] = {k: v.shape for k, v in aux3.items() if hasattr(v, "shape")}
                    self._save(current, basename, "member3_output")
                except Exception as e:
                    print(f"    Member3 failed: {e}")

        if self.config.use_member4:
            boxes = resolve_member4_boxes(
                image_path, basename,
                explicit=member4_boxes or self.config.member4_boxes,
                boxes_json=self.config.member4_boxes_json,
                boxes_dir=self.config.member4_boxes_dir,
            )
            if not boxes:
                print("  [Member4] Skipped: no eye/lip boxes (draw in GUI or pass --boxes_json).")
            else:
                print(f"  [Member4] Color/feature enhance ({len(boxes)} box(es))...")
                try:
                    out = enhance_beauty_Z(
                        current, boxes,
                        color_params=self.config.member4_color_params or ColorDetectParamsZ(),
                        enhance_params=self._member4_enhance_params(),
                    )
                    current = out.enhanced_rgb_u8
                    info.update(member4_skin_mask=out.skin_mask_u8, member4_eye_mask=out.eye_mask_u8,
                                member4_lip_mask=out.lip_mask_u8, member4_num_boxes=len(boxes))
                    self._save(current, basename, "member4_output")
                except Exception as e:
                    print(f"    Member4 failed: {e}")

        if self.config.use_member5_restoration:
            print(f"  [Member5] {self.config.member5_restoration_method} restoration...")
            current = member5_restoration(
                current, self.config.member5_restoration_method, self.config.member5_restoration_kernel_size
            )
            self._save(current, basename, "member5_output")

        info["output_shape"] = current.shape
        return current, info


# --- CLI ----------------------------------------------------------------------

def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        use_member2_pimple_removal=args.pimple_removal,
        use_member3=args.use_member3,
        use_member4=not args.no_member4,
        member4_whitening_strength=args.whitening,
        member4_boxes_json=args.boxes_json,
        member4_boxes_dir=args.boxes_dir,
        use_member5_restoration=args.restore,
        member5_restoration_method=args.restore_method,
        member5_restoration_kernel_size=args.kernel,
        output_dir=args.output,
        save_intermediate=args.save_intermediate,
    )


def run_on_image(
    img_path: str, config: PipelineConfig, *, basename: Optional[str] = None,
    member4_boxes: Optional[Sequence[RegionBox]] = None,
) -> np.ndarray:
    basename = basename or os.path.splitext(os.path.basename(img_path))[0]
    result, _ = CascadeBeautyPipeline(config).run(
        np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8),
        basename, image_path=img_path, member4_boxes=member4_boxes,
    )
    out_path = os.path.join(config.output_dir, f"{basename}_beauty.png")
    Image.fromarray(result).save(out_path)
    print(f"  Saved: {out_path}")
    return result


def run_on_folder(folder: str, config: PipelineConfig) -> None:
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if not files:
        print(f"No images in {folder}")
        return
    config.member4_boxes_dir = config.member4_boxes_dir or folder
    print("CASCADE BEAUTY PIPELINE — M3=%s M4=%s" % (config.use_member3, config.use_member4))
    for i, fname in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {fname}")
        run_on_image(os.path.join(folder, fname), config, basename=os.path.splitext(fname)[0])
    print("\nDONE.")


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
    from annotate_enhance_Z_ui import ImageAnnotatorView, _fit_inside

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root, self.input_path, self.output_img = root, None, None
            self._processing, self._pane_after = False, None
            root.title("Cascade Face Beauty Pipeline — eye/lip boxes")
            opts = ttk.LabelFrame(root, text="Pipeline options", padding=6)
            opts.pack(fill=tk.X, padx=8, pady=4)
            self.m2_var, self.m3_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
            self.m4_var, self.m5_var = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
            for txt, var in (
                ("M2: Pimple removal", self.m2_var),
                ("M3: Frequency skin smooth", self.m3_var),
                ("M4: Color & feature enhance (needs boxes)", self.m4_var),
                ("M5: Restoration", self.m5_var),
            ):
                ttk.Checkbutton(opts, text=txt, variable=var).pack(anchor=tk.W)
            rest = ttk.Frame(opts)
            rest.pack(anchor=tk.W, padx=16)
            ttk.Label(rest, text="Restore:").pack(side=tk.LEFT)
            self.method_var = tk.StringVar(value="mean")
            ttk.Combobox(rest, textvariable=self.method_var, values=["mean", "median", "gaussian", "vst"], width=9).pack(side=tk.LEFT, padx=4)
            self.kernel_var = tk.IntVar(value=3)
            ttk.Spinbox(rest, from_=1, to=7, width=4, textvariable=self.kernel_var, increment=2).pack(side=tk.LEFT)
            tools = ttk.LabelFrame(root, text="Eye / lip boxes (drag on image)", padding=6)
            tools.pack(fill=tk.X, padx=8, pady=2)
            for lbl, cmd in (
                ("Open image", self.open_image), ("Load boxes JSON", self.load_boxes),
                ("Save boxes JSON", self.save_boxes), ("Undo box", lambda: self.ann.undo_last_box()),
                ("Run pipeline", self.process_image), ("Save result", self.save_result),
            ):
                ttk.Button(tools, text=lbl, command=cmd).pack(side=tk.LEFT, padx=2)
            self.kind_var = tk.StringVar(value="eye")
            for k in ("eye", "lip"):
                ttk.Radiobutton(tools, text=k.title(), variable=self.kind_var, value=k, command=self._sync_kind).pack(side=tk.LEFT, padx=4)
            self.paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
            self.paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
            left = ttk.LabelFrame(self.paned, text="Annotate", padding=4)
            self.paned.add(left, weight=1)
            self.ann = ImageAnnotatorView(left, np.zeros((480, 640, 3), np.uint8))
            self.ann.pack(fill=tk.BOTH, expand=True)
            right = ttk.LabelFrame(self.paned, text="Pipeline output", padding=4)
            self.paned.add(right, weight=1)
            self.canvas = tk.Canvas(right, bg="#333", highlightthickness=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.paned.bind("<Configure>", lambda _e: self._sched_balance())
            self.status = ttk.Label(root, text="Open an image and draw eye/lip boxes, then Run pipeline.")
            self.status.pack(fill=tk.X, padx=8, pady=6)
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry(f"{int(sw * 0.9)}x{int(sh * 0.85)}")
            root.minsize(900, 600)
            root.after_idle(self._balance_panes)
            root.after(200, self._balance_panes)

        def _sched_balance(self) -> None:
            if self._pane_after:
                self.paned.after_cancel(self._pane_after)
            self._pane_after = self.paned.after(80, self._balance_panes)

        def _balance_panes(self) -> None:
            self._pane_after = None
            if self.paned.winfo_width() > 20:
                self.paned.sashpos(0, self.paned.winfo_width() // 2)

        def _sync_kind(self) -> None:
            self.ann.set_kind(self.kind_var.get())

        def _box_status(self) -> str:
            n_e = sum(1 for b in self.ann.boxes if b.kind == "eye")
            return f"boxes: eye={n_e} lip={sum(1 for b in self.ann.boxes if b.kind == 'lip')}"

        def open_image(self) -> None:
            if self._processing:
                return
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
            if not path:
                return
            self.input_path = path
            self.ann.load_image(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
            self.output_img = None
            self.canvas.delete("all")
            auto = default_boxes_json_path(path)
            if os.path.isfile(auto):
                try:
                    self.ann.boxes = load_boxes_json(auto)
                    self.ann._redraw()
                    self.status.config(text=f"Loaded {os.path.basename(path)} + {os.path.basename(auto)}")
                except Exception:
                    self.status.config(text=f"Loaded {os.path.basename(path)} (auto boxes failed)")
            else:
                self.status.config(text=f"Loaded {os.path.basename(path)} — {self._box_status()}")
            self.root.after(150, self.ann.fit_to_viewport)

        def load_boxes(self) -> None:
            p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if p:
                self.ann.boxes = load_boxes_json(p)
                self.ann._redraw()
                self.status.config(text=f"Boxes from {os.path.basename(p)} | {self._box_status()}")

        def save_boxes(self) -> None:
            if not self.input_path:
                return messagebox.showwarning("No image", "Open an image first.")
            if not self.ann.boxes:
                return messagebox.showwarning("No boxes", "Draw at least one eye or lip box.")
            p = filedialog.asksaveasfilename(
                defaultextension=".json", initialfile=os.path.basename(default_boxes_json_path(self.input_path))
            )
            if p:
                save_boxes_json(p, self.input_path, self.ann.boxes)
                self.status.config(text=f"Saved boxes: {os.path.basename(p)}")

        def _cfg(self) -> PipelineConfig:
            return PipelineConfig(
                use_member2_pimple_removal=self.m2_var.get(), use_member3=self.m3_var.get(),
                use_member4=self.m4_var.get(), use_member5_restoration=self.m5_var.get(),
                member5_restoration_method=self.method_var.get(),
                member5_restoration_kernel_size=self.kernel_var.get(),
                output_dir="./output_gui",
            )

        def process_image(self) -> None:
            if not self.input_path:
                return messagebox.showwarning("No image", "Open an image first.")
            if self.m4_var.get() and not self.ann.boxes:
                return messagebox.showwarning("No boxes", "Draw eye/lip boxes before running M4.")
            if self._processing:
                return
            self._processing = True
            self.status.config(text="Processing pipeline...")
            path, boxes, cfg = self.input_path, list(self.ann.boxes), self._cfg()

            def work() -> None:
                try:
                    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                    res, _ = CascadeBeautyPipeline(cfg).run(
                        rgb, os.path.splitext(os.path.basename(path))[0],
                        image_path=path, member4_boxes=boxes,
                    )
                    self.root.after(0, lambda: self._finish(res))
                except Exception as e:
                    self.root.after(0, lambda: self._error(str(e)))

            threading.Thread(target=work, daemon=True).start()

        def _finish(self, result: np.ndarray) -> None:
            self.output_img = result
            cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
            disp = Image.fromarray(result)
            _, dw, dh = _fit_inside(disp.width, disp.height, cw - 8, ch - 8)
            self._photo = ImageTk.PhotoImage(disp.resize((dw, dh), Image.Resampling.LANCZOS))
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)
            self.status.config(text=f"Done. {self._box_status()}")
            self._processing = False

        def _error(self, msg: str) -> None:
            messagebox.showerror("Pipeline error", msg)
            self.status.config(text="Error")
            self._processing = False

        def save_result(self) -> None:
            if self.output_img is None:
                return messagebox.showwarning("No result", "Run pipeline first.")
            p = filedialog.asksaveasfilename(defaultextension=".png", initialfile="beauty.png")
            if p:
                Image.fromarray(self.output_img).save(p)
                self.status.config(text=f"Saved {os.path.basename(p)}")

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Cascade M1→M2→M3→M4→M5 with eye/lip box UI")
    ap.add_argument("--input", type=str)
    ap.add_argument("--input_folder", type=str)
    ap.add_argument("--output", type=str, default=OUTPUT_DIR)
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--pimple_removal", action="store_true")
    ap.add_argument("--use_member3", action="store_true")
    ap.add_argument("--no_member4", action="store_true")
    ap.add_argument("--whitening", type=float, default=0.85)
    ap.add_argument("--boxes_json", type=str)
    ap.add_argument("--boxes_dir", type=str)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--restore_method", type=str, default="mean", choices=["mean", "median", "gaussian", "vst"])
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--show_gui", action="store_true")
    args = ap.parse_args()
    if args.show_gui:
        launch_gui()
        return
    cfg = _config_from_args(args)
    os.makedirs(args.output, exist_ok=True)
    if args.input:
        run_on_image(args.input, cfg)
    elif args.input_folder:
        if args.boxes_dir:
            cfg.member4_boxes_dir = args.boxes_dir
        run_on_folder(args.input_folder, cfg)
    else:
        print("Provide --input, --input_folder, or --show_gui")


if __name__ == "__main__":
    main()
