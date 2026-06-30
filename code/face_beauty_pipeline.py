"""
face_beauty_pipeline.py
Member 5: Restoration, Integration, Evaluation & Report

Integrates:
- Member 1: Segmentation (1_segmentation.py)
- Member 2: Morphology (2_morphology.py)
- Member 3: Skin smoothing + frequency enhancement (skin_smooth_freq_enhance.py)
- Member 4: Color & Feature Enhancement (4_color_feature_enhance.py)
- Additional restoration for noise/blemish removal

Usage examples:
    python face_beauty_pipeline.py --input image.png --use_member3
    python face_beauty_pipeline.py --input_folder ./dataset --use_member3 --use_member4_after
    python face_beauty_pipeline.py --show_gui
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage
import threading

OUTPUT_DIR = "./output/output_member5"

# ========================
# IMPORT MEMBER MODULES
# ========================

# Member 1
from segmentation import (
    rgb_to_ycbcr,
    ycbcr_skin_mask,
    otsu_threshold,
    segment_v4,
    read_rgb,
    save_image,
    CB_LOW, CB_HIGH, CR_LOW, CR_HIGH,
    binary_dilate,
    binary_erode,
    refine_mask
)

# Member 2
from morphology import (
    MorphologyRefiner,
    KERNEL_SIZE as MORPH_KERNEL_SIZE,
    MIN_OBJECT_SIZE
)

# Member 3: Skin smoothing + frequency enhancement
try:
    from skin_smooth_freq_enhance import process_face_skin_smooth_and_freq_enhance
    MEMBER3_AVAILABLE = True
except ImportError:
    MEMBER3_AVAILABLE = False
    print("Warning: Member 3 module (skin_smooth_freq_enhance.py) not found. Disabling member3 features.")

# Member 4
from color_feature_enhance import (
    rgb_to_hsv_u01,
    hsv_u01_to_rgb_u8,
    rgb_to_ycbcr_u8,
    enhance_contrast_rgb_u8,
    ycbcr_skin_mask_u8_adaptive,
    adjust_skin_tone_rgb,
    adjust_skin_tone_hsv,
    skin_tone_adjust,
    SkinToneParams,
    whiten_and_smooth_skin,
    enhance_eyes_and_lips,
    FeatureEnhanceParams,
    apply_pseudocolor,
    estimate_face_regions_from_skin_mask,
    estimate_eye_lip_masks_u8,
    FaceRegions,
    enhance_beauty,
    EnhanceOutputs,
    _connected_components_bboxes,
    _boost_soft_alpha,
    _suppress_highlights_alpha,
    _read_rgb as read_rgb_member4,
    _save_rgb as save_rgb_member4,
    _save_mask as save_mask_member4,
)

# ========================
# RESTORATION MODULE
# ========================

def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean filter for general noise reduction."""
    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = ndimage.uniform_filter(image[:, :, c].astype(np.float32), size=size)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = ndimage.uniform_filter(image.astype(np.float32), size=size)
        return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter for salt-and-pepper noise removal."""
    if image.ndim == 3:
        result = np.zeros_like(image)
        for c in range(3):
            result[:, :, c] = ndimage.median_filter(image[:, :, c], size=size)
        return result
    else:
        return ndimage.median_filter(image, size=size)


def gaussian_filter(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian filter for general noise reduction."""
    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = ndimage.gaussian_filter(image[:, :, c].astype(np.float32), sigma=sigma)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
        return np.clip(result, 0, 255).astype(np.uint8)


def create_motion_psf(length: int, angle: float) -> np.ndarray:
    """
    Create motion blur PSF (similar to MATLAB's fspecial('motion'))
    """
    size = length
    psf = np.zeros((size, size))
    center = (size - 1) / 2
    rad = np.deg2rad(angle)
    x = np.arange(size) - center
    y = np.arange(size) - center
    xx, yy = np.meshgrid(x, y)
    dist = np.abs(xx * np.sin(rad) - yy * np.cos(rad))
    on_line = dist < 0.5
    if on_line.sum() > 0:
        psf[on_line] = 1.0 / on_line.sum()
    return psf


def wiener_filter_2d(image: np.ndarray, kernel_size: int = 5, noise_power: float = 0.01) -> np.ndarray:
    """
    Wiener filter - standard implementation (Gaussian PSF).
    """
    if image.ndim == 3:
        restored_channels = []
        for c in range(image.shape[2]):
            restored_c = wiener_filter_2d(image[:, :, c], kernel_size, noise_power)
            restored_channels.append(restored_c)
        return np.stack(restored_channels, axis=2)

    img_float = image.astype(np.float32) / 255.0
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    sigma = k / 3.0
    ax = np.linspace(-(k-1)//2, (k-1)//2, k)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    H = np.fft.fft2(np.fft.ifftshift(kernel), s=img_float.shape)
    G = np.fft.fft2(img_float)

    signal_var = np.var(img_float)
    NSR = noise_power / (signal_var + 1e-8)

    H_conj = np.conj(H)
    H_sq = np.abs(H) ** 2
    denominator = H_sq + NSR
    denominator = np.maximum(denominator, 1e-8)

    restored = np.fft.ifft2(H_conj / denominator * G)
    restored = np.abs(restored)
    restored = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def restore_blemishes(rgb: np.ndarray, method: str = "mean", kernel_size: int = 3) -> np.ndarray:
    """Remove blemishes using various filtering methods."""
    method = method.lower()
    if method == "mean":
        return mean_filter(rgb, size=kernel_size)
    elif method == "median":
        return median_filter(rgb, size=kernel_size)
    elif method == "gaussian":
        return gaussian_filter(rgb, sigma=kernel_size/2)
    elif method == "wiener":
        return wiener_filter_2d(rgb, kernel_size=kernel_size, noise_power=0.01)
    else:
        print(f"Unknown method '{method}', using mean filter")
        return mean_filter(rgb, size=kernel_size)


# ========================
# SIMPLE SKIN SMOOTHING (FALLBACK)
# ========================

def apply_skin_smoothing(rgb: np.ndarray, skin_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Fallback simple Gaussian smoothing on skin."""
    result = rgb.copy().astype(np.float32)
    mask_2d = skin_mask > 127
    if not np.any(mask_2d):
        return rgb
    for c in range(3):
        channel_smooth = ndimage.gaussian_filter(rgb[:, :, c].astype(np.float32), sigma=sigma)
        channel_result = result[:, :, c]
        channel_result[mask_2d] = channel_smooth[mask_2d]
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_skin_tone_simple(rgb: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """Simple warm tone adjustment."""
    result = rgb.copy().astype(np.float32)
    mask_2d = skin_mask > 127
    if not np.any(mask_2d):
        return rgb
    gain = np.array([1.05, 1.02, 0.98])
    bias = np.array([5, 3, 0])
    for c in range(3):
        result_channel = result[:, :, c]
        result_channel[mask_2d] = result_channel[mask_2d] * gain[c] + bias[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_eyes_lips_simple(rgb: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """Simple eyes/lips enhancement based on face geometry."""
    h, w = rgb.shape[:2]
    ys, xs = np.where(skin_mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return rgb
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    face_h = y1 - y0
    face_w = x1 - x0

    eye_y0 = max(0, y0 + int(0.25 * face_h))
    eye_y1 = min(h, y0 + int(0.45 * face_h))
    eye_x0 = max(0, x0 + int(0.2 * face_w))
    eye_x1 = min(w, x1 - int(0.2 * face_w))

    lip_y0 = max(0, y0 + int(0.65 * face_h))
    lip_y1 = min(h, y0 + int(0.85 * face_h))
    lip_x0 = max(0, x0 + int(0.25 * face_w))
    lip_x1 = min(w, x1 - int(0.25 * face_w))

    hsv = rgb_to_hsv_u01(rgb)
    h_plane, s_plane, v_plane = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    eye_mask = np.zeros((h, w), dtype=bool)
    eye_mask[eye_y0:eye_y1, eye_x0:eye_x1] = True
    eye_mask = eye_mask & (skin_mask > 127)
    if np.any(eye_mask):
        v_plane[eye_mask] = np.clip(v_plane[eye_mask] * 1.2, 0, 1)
        s_plane[eye_mask] = np.clip(s_plane[eye_mask] * 0.9, 0, 1)

    lip_mask = np.zeros((h, w), dtype=bool)
    lip_mask[lip_y0:lip_y1, lip_x0:lip_x1] = True
    lip_mask = lip_mask & (skin_mask > 127)
    if np.any(lip_mask):
        s_plane[lip_mask] = np.clip(s_plane[lip_mask] * 1.4, 0, 1)
        v_plane[lip_mask] = np.clip(v_plane[lip_mask] * 1.05, 0, 1)
        h_plane[lip_mask] = (h_plane[lip_mask] + 5.0 / 360.0) % 1.0

    enhanced = hsv_u01_to_rgb_u8(np.stack([h_plane, s_plane, v_plane], axis=-1))
    return enhanced


# ========================
# MAIN BEAUTY PIPELINE
# ========================

@dataclass
class PipelineConfig:
    """Configuration for the beauty enhancement pipeline."""
    enable_restoration: bool = False
    restoration_method: str = "mean"      # 'mean', 'median', 'gaussian', 'wiener'
    restoration_kernel_size: int = 3

    # Member 3 options
    use_member3: bool = False             # Enable Member 3 skin smoothing + frequency enhancement
    use_member4_after_member3: bool = False  # Apply Member 4 color/feature enhancement after Member 3

    # Member 4 options
    use_member4_enhancement: bool = True  # Use Member 4's advanced enhancement (if use_member3=False or after)
    enable_skin_smoothing: bool = True
    enable_skin_tone: bool = True
    enable_eyes_lips: bool = True
    smoothing_sigma: float = 2.0
    whitening_strength: float = 0.85

    output_dir: str = "./output/output_member5"


class FaceBeautyPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._ensure_output_dir()
        self.morphology_refiner = MorphologyRefiner(kernel_size=MORPH_KERNEL_SIZE)

    def _ensure_output_dir(self):
        os.makedirs(self.config.output_dir, exist_ok=True)

    def get_skin_mask(self, rgb: np.ndarray) -> np.ndarray:
        mask_final, info = segment_v4(rgb)
        refined_mask = self.morphology_refiner.refine(mask_final)
        return refined_mask

    def process_image(self, rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {"original_shape": rgb.shape}
        current = rgb.copy()

        # Step 1: Restoration (optional)
        if self.config.enable_restoration:
            print(f"  Applying {self.config.restoration_method} restoration...")
            current = restore_blemishes(
                current,
                method=self.config.restoration_method,
                kernel_size=self.config.restoration_kernel_size
            )

        # Step 2: Skin mask (needed for later steps)
        print("  Generating skin mask...")
        skin_mask = self.get_skin_mask(current)
        info["skin_mask_shape"] = skin_mask.shape

        # ========================
        # Step 3: Member 3 - Skin smoothing + frequency enhancement
        # ========================
        if self.config.use_member3:
            if not MEMBER3_AVAILABLE:
                print("  Member 3 module not available, skipping.")
            else:
                print("  Applying Member 3 skin smooth + frequency enhancement...")
                try:
                    # Member3 function expects uint8 RGB, returns (final_rgb, aux_dict)
                    current, aux3 = process_face_skin_smooth_and_freq_enhance(
                        current,
                        avg_radius=2,
                        gauss_sigma=1.2,
                        fft_smooth_sigma_norm=0.07,
                        spatial_mix=(0.35, 0.35, 0.30),
                        freq_boost_sigma_norm=0.055,
                        freq_boost_amount=0.55,
                        feather_radius=14,
                        skin_strength=0.92,
                        boost_strength=0.65,
                    )
                    info["member3_aux"] = {k: v.shape for k, v in aux3.items() if hasattr(v, 'shape')}
                except Exception as e:
                    print(f"  Member 3 processing failed: {e}")

        # ========================
        # Step 4: Member 4 (if not disabled, and either not using member3 or using after)
        # ========================
        if self.config.use_member4_enhancement:
            # If member3 is active and we do NOT want member4 after, skip member4.
            if self.config.use_member3 and not self.config.use_member4_after_member3:
                print("  Skipping Member 4 enhancement (use_member4_after_member3=False).")
            else:
                print("  Applying Member 4 enhancement pipeline...")
                outputs = enhance_beauty(
                    current,
                    skin_mask_u8=skin_mask,
                    skin_params=SkinToneParams(),
                    feature_params=FeatureEnhanceParams(
                        eye_v_gamma=0.70,
                        eye_unsharp_radius=5,
                        eye_unsharp_amount=1.65,
                        eye_s_gain=0.85,
                        lip_v_gamma=0.95,
                        lip_unsharp_radius=4,
                        lip_unsharp_amount=1.25,
                        lip_s_gain=1.55,
                        lip_hue_shift=0.0,
                    ),
                    pseudocolor_cmap="jet",
                )
                current = outputs.enhanced_rgb_u8
                info["skin_mask"] = outputs.skin_mask_u8
                info["eye_mask"] = outputs.eye_mask_u8
                info["lip_mask"] = outputs.lip_mask_u8

        info["output_shape"] = current.shape
        return current, info

    def process_folder(self, input_folder: str, output_folder: Optional[str] = None):
        if output_folder:
            self.config.output_dir = output_folder
            self._ensure_output_dir()

        extensions = (".png", ".jpg", ".jpeg", ".bmp")
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]
        if not files:
            print(f"No image files found in {input_folder}")
            return

        print("=" * 60)
        print("MEMBER 5: INTEGRATED FACE BEAUTY PIPELINE")
        print(f"Processing {len(files)} images from {input_folder}")
        print(f"Restoration: {self.config.enable_restoration} ({self.config.restoration_method})")
        print(f"Use Member 3: {self.config.use_member3}")
        if self.config.use_member3:
            print(f"  Member 4 after Member 3: {self.config.use_member4_after_member3}")
        print(f"Use Member 4 Enhancement: {self.config.use_member4_enhancement}")
        if not self.config.use_member4_enhancement and not self.config.use_member3:
            print("  No enhancement selected! Only restoration (if enabled) will be applied.")
        print(f"Output folder: {self.config.output_dir}")
        print("=" * 60)

        for i, filename in enumerate(sorted(files), 1):
            print(f"\n[{i}/{len(files)}] Processing {filename}...")
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

            final_img, info = self.process_image(rgb)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_name = os.path.splitext(filename)[0] + "_beauty.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            Image.fromarray(final_img).save(out_path)

            if self.config.use_member4_enhancement and "eye_mask" in info:
                eye_name = os.path.splitext(filename)[0] + "_eye_mask.png"
                lip_name = os.path.splitext(filename)[0] + "_lip_mask.png"
                Image.fromarray(info["eye_mask"]).save(os.path.join(self.config.output_dir, eye_name))
                Image.fromarray(info["lip_mask"]).save(os.path.join(self.config.output_dir, lip_name))

            print(f"  ✓ Saved to {out_path}")

        print("\n" + "=" * 60)
        print(f"DONE! Processed {len(files)} images.")
        print(f"Output directory: {self.config.output_dir}")
        print("=" * 60)


# ========================
# SIMPLE GUI (with Member 3 options)
# ========================

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
    import threading

    class BeautyApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Face Beauty Pipeline - Member 5")
            self.root.geometry("900x800")

            self.input_path = None
            self.output_img = None
            self._processing = False

            # 控制面板
            control_frame = tk.Frame(root)
            control_frame.pack(pady=10)

            self.btn_open = tk.Button(control_frame, text="📁 Open Image", command=self.open_image,
                                      bg="lightblue", width=12)
            self.btn_open.pack(side=tk.LEFT, padx=5)

            self.btn_process = tk.Button(control_frame, text="✨ Process", command=self.process_image,
                                         bg="lightgreen", width=12)
            self.btn_process.pack(side=tk.LEFT, padx=5)

            self.btn_save = tk.Button(control_frame, text="💾 Save Result", command=self.save_result,
                                      bg="lightyellow", width=12)
            self.btn_save.pack(side=tk.LEFT, padx=5)

            # 选项框架
            options_frame = tk.LabelFrame(root, text="Processing Options", padx=10, pady=5)
            options_frame.pack(pady=5, fill=tk.X)

            self.use_m3_var = tk.BooleanVar(value=False)
            tk.Checkbutton(options_frame, text="Use Member 3 (Skin smooth + Frequency enhance)",
                          variable=self.use_m3_var).pack(anchor=tk.W, padx=10, pady=2)

            self.m3_then_m4_var = tk.BooleanVar(value=True)
            tk.Checkbutton(options_frame, text="Apply Member 4 after Member 3",
                          variable=self.m3_then_m4_var).pack(anchor=tk.W, padx=30, pady=2)

            self.use_m4_var = tk.BooleanVar(value=True)
            tk.Checkbutton(options_frame, text="Use Member 4 (Color + Eye/Lip enhancement)",
                          variable=self.use_m4_var).pack(anchor=tk.W, padx=10, pady=2)

            self.restore_var = tk.BooleanVar(value=False)
            tk.Checkbutton(options_frame, text="Enable Restoration",
                          variable=self.restore_var).pack(anchor=tk.W, padx=10, pady=2)

            rest_frame = tk.Frame(options_frame)
            rest_frame.pack(anchor=tk.W, padx=30, pady=2)
            tk.Label(rest_frame, text="Method:").pack(side=tk.LEFT)
            self.method_var = tk.StringVar(value="mean")
            method_menu = ttk.Combobox(rest_frame, textvariable=self.method_var,
                                       values=["mean", "median", "gaussian", "wiener"], width=10)
            method_menu.pack(side=tk.LEFT, padx=5)
            tk.Label(rest_frame, text="Kernel:").pack(side=tk.LEFT, padx=5)
            self.kernel_var = tk.IntVar(value=3)
            tk.Spinbox(rest_frame, from_=1, to=7, width=5, textvariable=self.kernel_var, increment=2).pack(side=tk.LEFT)

            # 图像显示
            display_frame = tk.Frame(root)
            display_frame.pack(pady=10, expand=True, fill=tk.BOTH)
            left_frame = tk.Frame(display_frame)
            left_frame.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.BOTH)
            tk.Label(left_frame, text="Original", font=("Arial", 10, "bold")).pack()
            self.canvas_orig = tk.Canvas(left_frame, bg="gray", width=400, height=400)
            self.canvas_orig.pack(expand=True, fill=tk.BOTH)
            right_frame = tk.Frame(display_frame)
            right_frame.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.BOTH)
            tk.Label(right_frame, text="Processed", font=("Arial", 10, "bold")).pack()
            self.canvas_result = tk.Canvas(right_frame, bg="gray", width=400, height=400)
            self.canvas_result.pack(expand=True, fill=tk.BOTH)

            self.status_label = tk.Label(root, text="Ready - Open an image", fg="blue")
            self.status_label.pack(pady=5)

        def set_buttons_state(self, state):
            """启用/禁用所有操作按钮（处理期间避免干扰）"""
            self.btn_open.config(state=state)
            self.btn_process.config(state=state)
            self.btn_save.config(state=state)

        def open_image(self):
            if self._processing:
                messagebox.showinfo("Busy", "Please wait, processing in progress...")
                return
            path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if path:
                self.input_path = path
                img = Image.open(path)
                img.thumbnail((400, 400))
                self.original_photo = ImageTk.PhotoImage(img)
                self.canvas_orig.delete("all")
                self.canvas_orig.create_image(200, 200, image=self.original_photo, anchor=tk.CENTER)
                self.status_label.config(text=f"Loaded: {os.path.basename(path)}", fg="green")
                self.output_img = None
                self.canvas_result.delete("all")

        def process_image(self):
            if not self.input_path:
                messagebox.showwarning("No Image", "Please open an image first.")
                return
            if self._processing:
                messagebox.showinfo("Busy", "Already processing, please wait...")
                return

            self._processing = True
            self.set_buttons_state(tk.DISABLED)
            self.status_label.config(text="Processing... (please wait)", fg="orange")
            self.root.update()

            # 启动后台线程
            thread = threading.Thread(target=self._process_thread, daemon=True)
            thread.start()

        def _process_thread(self):
            try:
                with Image.open(self.input_path) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

                config = PipelineConfig(
                    enable_restoration=self.restore_var.get(),
                    restoration_method=self.method_var.get(),
                    restoration_kernel_size=self.kernel_var.get(),
                    use_member3=self.use_m3_var.get(),
                    use_member4_after_member3=self.m3_then_m4_var.get(),
                    use_member4_enhancement=self.use_m4_var.get(),
                    enable_skin_smoothing=True,
                    enable_skin_tone=True,
                    enable_eyes_lips=True,
                )
                pipeline = FaceBeautyPipeline(config)
                result, _ = pipeline.process_image(rgb)

                # 将结果传回主线程
                self.root.after(0, self._finish_processing, result)
            except Exception as e:
                self.root.after(0, self._show_error, str(e))

        def _finish_processing(self, result):
            self.output_img = result
            display = Image.fromarray(result)
            display.thumbnail((400, 400))
            self.result_photo = ImageTk.PhotoImage(display)
            self.canvas_result.delete("all")
            self.canvas_result.create_image(200, 200, image=self.result_photo, anchor=tk.CENTER)
            self.status_label.config(text="Complete! ✓", fg="green")
            self._processing = False
            self.set_buttons_state(tk.NORMAL)

        def _show_error(self, err_msg):
            messagebox.showerror("Processing Error", f"Failed to process image:\n{err_msg}")
            self.status_label.config(text="Error occurred", fg="red")
            self._processing = False
            self.set_buttons_state(tk.NORMAL)

        def save_result(self):
            if self._processing:
                messagebox.showinfo("Busy", "Please wait, processing in progress...")
                return
            if self.output_img is None:
                messagebox.showwarning("No Result", "Process an image first.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                Image.fromarray(self.output_img).save(path)
                self.status_label.config(text=f"Saved to {os.path.basename(path)}", fg="blue")

    root = tk.Tk()
    app = BeautyApp(root)
    root.mainloop()


# ========================
# COMMAND LINE INTERFACE
# ========================

def main():
    parser = argparse.ArgumentParser(description="Face Beauty Pipeline - Member 5 (with Member 3 integration)")
    parser.add_argument("--input", type=str, help="Single image file")
    parser.add_argument("--input_folder", type=str, help="Folder with images")
    parser.add_argument("--output", type=str, default="./output_member5", help="Output folder")
    parser.add_argument("--restore", action="store_true", help="Enable restoration")
    parser.add_argument("--restore_method", type=str, default="mean",
                       choices=["mean", "median", "gaussian", "wiener"],
                       help="Restoration method")
    parser.add_argument("--kernel", type=int, default=3, help="Filter kernel size")

    # Member 3 options
    parser.add_argument("--use_member3", action="store_true", help="Use Member 3 skin smooth + freq enhance")
    parser.add_argument("--member4_after_member3", action="store_true", default=True,
                       help="Apply Member 4 after Member 3 (default True)")

    # Member 4 options
    parser.add_argument("--simple", action="store_true", help="Disable Member 4 advanced enhancement (fallback to simple)")
    parser.add_argument("--show_gui", action="store_true", help="Launch GUI")

    args = parser.parse_args()

    if args.show_gui:
        launch_gui()
        return

    config = PipelineConfig(
        enable_restoration=args.restore,
        restoration_method=args.restore_method,
        restoration_kernel_size=args.kernel,
        use_member3=args.use_member3,
        use_member4_after_member3=args.member4_after_member3,
        use_member4_enhancement=not args.simple,
        output_dir=args.output
    )
    pipeline = FaceBeautyPipeline(config)

    if args.input:
        print("Processing single image...")
        with Image.open(args.input) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        result, _ = pipeline.process_image(rgb)
        out_path = os.path.join(args.output, "result_beauty.png")
        os.makedirs(args.output, exist_ok=True)
        Image.fromarray(result).save(out_path)
        print(f"Saved to {out_path}")
    elif args.input_folder:
        pipeline.process_folder(args.input_folder, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()