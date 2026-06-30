"""Skin enhancement via frequency separation (low / high layers).

- **Low-frequency layer**: Gaussian base (tone, shading). Smoothed inside skin mask.
- **High-frequency layer**: Original − Low (pores, fine lines, edges). Kept intact and
  recombined after low-layer processing: ``output = low_processed + high``.

Uses segmentation.segment_v4 for a skin mask. Run from repo ``code`` folder::

    python skin_smooth_freq_enhance.py --input ../dataset/face.png --output ../output/skin_freq/

Library::

    from skin_smooth_freq_enhance import process_face_skin_smooth_and_freq_enhance
    out, info = process_face_skin_smooth_and_freq_enhance(rgb_uint8)
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from segmentation import read_rgb, save_image, segment_v4

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover
    _ndimage = None


def _box_blur_gray_f32(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gray.astype(np.float32, copy=False)
    h, w = gray.shape
    r = int(radius)
    padded = np.pad(gray.astype(np.float32), ((r, r), (r, r)), mode="edge")
    integ = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float32)
    integ[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    y0 = np.arange(0, h, dtype=np.int32)
    y1 = y0 + (2 * r + 1)
    x0 = np.arange(0, w, dtype=np.int32)
    x1 = x0 + (2 * r + 1)
    a = integ[y0[:, None], x0[None, :]]
    b = integ[y0[:, None], x1[None, :]]
    c = integ[y1[:, None], x0[None, :]]
    d = integ[y1[:, None], x1[None, :]]
    area = float((2 * r + 1) ** 2)
    return (d - b - c + a) / area


def average_filter_rgb(rgb_u8: np.ndarray, kernel_radius: int = 2) -> np.ndarray:
    x = rgb_u8.astype(np.float32)
    out = np.empty_like(x)
    for c in range(3):
        out[:, :, c] = _box_blur_gray_f32(x[:, :, c], radius=kernel_radius)
    return np.clip(out, 0, 255).astype(np.uint8)


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    sigma = float(max(sigma, 1e-6))
    r = max(int(truncate * sigma + 0.5), 1)
    t = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(t * t) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def _conv1d_horiz(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.astype(np.float32)
    n = int(k.size)
    pad = n // 2
    g = gray.astype(np.float32, copy=False)
    p = np.pad(g, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(g, dtype=np.float32)
    for i in range(n):
        out += float(k[i]) * p[:, i : i + g.shape[1]]
    return out


def _conv1d_vert(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.astype(np.float32)
    n = int(k.size)
    pad = n // 2
    g = gray.astype(np.float32, copy=False)
    p = np.pad(g, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(g, dtype=np.float32)
    for i in range(n):
        out += float(k[i]) * p[i : i + g.shape[0], :]
    return out


def gaussian_filter_rgb_f32(rgb_f32: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian on float RGB (0..255 linear)."""
    kh = _gaussian_kernel_1d(sigma)
    x = rgb_f32.astype(np.float32, copy=False)
    out = np.empty_like(x)
    for c in range(3):
        g = x[:, :, c]
        out[:, :, c] = _conv1d_vert(_conv1d_horiz(g, kh), kh)
    return out


def gaussian_filter_rgb(rgb_u8: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    return np.clip(gaussian_filter_rgb_f32(rgb_u8.astype(np.float32), sigma), 0, 255).astype(
        np.uint8
    )


def decompose_frequency_layers(
    rgb_f32: np.ndarray,
    split_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into low / high frequency layers (linear RGB).

    - ``low``  = Gaussian blur (tone, large-scale shading)
    - ``high`` = ``rgb - low`` (texture, pores, edges)
    """
    low = gaussian_filter_rgb_f32(rgb_f32, sigma=float(split_sigma))
    high = rgb_f32 - low
    return low, high


def recompose_frequency_layers(
    low_f32: np.ndarray,
    high_f32: np.ndarray,
    *,
    high_gain: float = 1.0,
) -> np.ndarray:
    """Recombine layers; ``high_gain`` scales texture strength (1.0 = unchanged)."""
    out = low_f32 + float(high_gain) * high_f32
    return np.clip(out, 0, 255).astype(np.uint8)


def process_low_frequency_layer(
    low_f32: np.ndarray,
    alpha_u01: np.ndarray,
    *,
    avg_radius: int = 2,
    gauss_sigma: float = 0.7,
    fft_smooth_sigma_norm: float = 0.07,
    spatial_mix: Tuple[float, float, float] = (0.40, 0.22, 0.38),
) -> np.ndarray:
    """
    Smooth only the low-frequency layer inside the skin mask (box + Gaussian + FFT LP).
    Non-skin pixels keep the original low layer.
    """
    wa, wg, wf = spatial_mix
    s = wa + wg + wf
    if s <= 0:
        wa, wg, wf = 1.0, 1.0, 1.0
        s = 3.0
    wa, wg, wf = wa / s, wg / s, wf / s

    low_u8 = np.clip(low_f32, 0, 255).astype(np.uint8)
    avg_b = average_filter_rgb(low_u8, kernel_radius=avg_radius).astype(np.float32)
    gauss_b = gaussian_filter_rgb_f32(low_f32, sigma=gauss_sigma)
    fft_lp_b = fft_gaussian_lowpass_rgb(low_u8, sigma_norm=fft_smooth_sigma_norm).astype(
        np.float32
    )
    smooth_low = wa * avg_b + wg * gauss_b + wf * fft_lp_b

    a = np.clip(alpha_u01[..., None], 0.0, 1.0).astype(np.float32)
    return low_f32 * (1.0 - a) + smooth_low * a


@lru_cache(maxsize=32)
def _fft_gauss_mask(h: int, w: int, sigma_norm: float) -> np.ndarray:
    """Cached shifted Gaussian low-pass mask, shape (H, W)."""
    fy = np.fft.fftfreq(h)[:, None].astype(np.float32)
    fx = np.fft.fftfreq(w)[None, :].astype(np.float32)
    rr2 = np.fft.fftshift(fx) ** 2 + np.fft.fftshift(fy) ** 2
    sn = float(max(sigma_norm, 1e-8))
    return np.exp(-rr2 / (2.0 * sn * sn)).astype(np.float32)


def _fft_filter_rgb_f32(
    rgb_f01: np.ndarray,
    spectral_mask_hw: np.ndarray,
    *,
    boost_amount: float = 0.0,
) -> np.ndarray:
    """
  Apply a shifted spectral mask to RGB in [0,1].

  ``boost_amount == 0``: low-pass. Otherwise one-pass high-boost in frequency domain:
  ``out = x + amount * (x - lowpass(x))`` equivalent to multiplying spectrum by
  ``1 + amount * (1 - mask)``.
    """
    f = np.fft.fft2(rgb_f01, axes=(-2, -1))
    f_s = np.fft.fftshift(f, axes=(-2, -1))
    m = spectral_mask_hw
    if boost_amount != 0.0:
        filt = 1.0 + float(boost_amount) * (1.0 - m)
        f_s = f_s * filt[..., None]
    else:
        f_s = f_s * m[..., None]
    out = np.real(
        np.fft.ifft2(np.fft.ifftshift(f_s, axes=(-2, -1)), axes=(-2, -1))
    )
    return out


def fft_gaussian_lowpass_gray(gray_u8: np.ndarray, sigma_norm: float = 0.08) -> np.ndarray:
    """FFT-domain Gaussian low-pass; ``sigma_norm`` scales cutoff on shifted freq grid."""
    g = gray_u8.astype(np.float32) / 255.0
    h, w = g.shape
    mask = _fft_gauss_mask(h, w, float(sigma_norm))
    out = _fft_filter_rgb_f32(g[..., None], mask)[:, :, 0]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def fft_gaussian_lowpass_rgb(rgb_u8: np.ndarray, sigma_norm: float = 0.08) -> np.ndarray:
    g = rgb_u8.astype(np.float32) / 255.0
    h, w = g.shape[:2]
    mask = _fft_gauss_mask(h, w, float(sigma_norm))
    out = _fft_filter_rgb_f32(g, mask)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def fft_highboost_rgb(rgb_u8: np.ndarray, sigma_norm: float = 0.06, amount: float = 0.85) -> np.ndarray:
    """``out = x + amount * (x - lowpass_fft(x))`` — single FFT round-trip on RGB."""
    x = rgb_u8.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    mask = _fft_gauss_mask(h, w, float(sigma_norm))
    boosted = _fft_filter_rgb_f32(x, mask, boost_amount=float(amount))
    return np.clip(boosted * 255.0, 0, 255).astype(np.uint8)


def feather_mask_u01(mask_u8: np.ndarray, radius: int = 12) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if radius <= 0:
        return m
    a = _box_blur_gray_f32(m, radius=max(1, int(radius)))
    return np.clip(a, 0.0, 1.0)


def blend_skin(
    original_u8: np.ndarray,
    processed_u8: np.ndarray,
    alpha_u01: np.ndarray,
) -> np.ndarray:
    """Blend with ``alpha_u01`` in [0,1], shape (H,W)."""
    a = np.clip(alpha_u01[..., None], 0.0, 1.0).astype(np.float32)
    o = original_u8.astype(np.float32)
    p = processed_u8.astype(np.float32)
    return np.clip(o * (1.0 - a) + p * a, 0, 255).astype(np.uint8)


def _box_blur_rgb_f32(rgb_f32: np.ndarray, radius: int) -> np.ndarray:
    out = np.empty_like(rgb_f32, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = _box_blur_gray_f32(rgb_f32[:, :, c], radius=max(1, int(radius)))
    return out


def _high_magnitude_f32(high_f32: np.ndarray) -> np.ndarray:
    """Per-pixel max |channel| of the high-frequency residual."""
    return np.max(np.abs(high_f32), axis=2)


def _filter_spot_components_u8(
    cand: np.ndarray,
    *,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    if not np.any(cand):
        return np.zeros(cand.shape, dtype=np.uint8)
    if _ndimage is None:
        return (cand.astype(np.uint8) * 255)
    labeled, n = _ndimage.label(cand)
    if n == 0:
        return np.zeros(cand.shape, dtype=np.uint8)
    sizes = _ndimage.sum(cand, labeled, index=range(1, n + 1))
    out = np.zeros(cand.shape, dtype=bool)
    for i, size in enumerate(sizes, start=1):
        if min_area <= size <= max_area:
            out[labeled == i] = True
    return (out.astype(np.uint8) * 255)


def detect_high_spike_mask_u8(
    high_f32: np.ndarray,
    skin_mask_u8: np.ndarray,
    *,
    spike_radius: int = 3,
    spike_threshold: float = 7.0,
    rel_ratio: float = 1.45,
    min_area: int = 2,
    max_area: int = 350,
) -> np.ndarray:
    """
    Detect isolated spikes on the high-frequency magnitude map.

    A pixel is a spike when ``|high|`` exceeds its local neighborhood (box blur)
    by ``spike_threshold`` or by factor ``rel_ratio``.
    """
    skin = skin_mask_u8 > 127
    if not np.any(skin):
        return np.zeros(skin_mask_u8.shape[:2], dtype=np.uint8)

    mag = _high_magnitude_f32(high_f32)
    r = max(1, int(spike_radius))
    mag_lp = _box_blur_gray_f32(mag, radius=r)
    excess = mag - mag_lp
    rr = float(max(rel_ratio, 1.01))
    thr = float(spike_threshold)
    spike = (excess > thr) | (mag > mag_lp * rr + 1.5)
    cand = skin & spike
    return _filter_spot_components_u8(
        cand, min_area=int(min_area), max_area=int(max_area)
    )


def remove_acne_on_high_layer(
    high_f32: np.ndarray,
    skin_mask_u8: np.ndarray,
    skin_alpha_u01: np.ndarray,
    *,
    spike_radius: int = 3,
    spike_threshold: float = 7.0,
    rel_ratio: float = 1.45,
    min_area: int = 2,
    max_area: int = 350,
    feather_radius: int = 4,
    strength: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-frequency blemish removal: spike detection + press toward zero.

    ``high_out = high * (1 - alpha)`` inside detected spikes (alpha feathered on skin).
    """
    spike_u8 = detect_high_spike_mask_u8(
        high_f32,
        skin_mask_u8,
        spike_radius=spike_radius,
        spike_threshold=spike_threshold,
        rel_ratio=rel_ratio,
        min_area=min_area,
        max_area=max_area,
    )
    if not np.any(spike_u8 > 0):
        return high_f32, spike_u8

    acne_a = feather_mask_u01(spike_u8, radius=max(1, int(feather_radius)))
    acne_a = np.clip(acne_a * float(strength), 0.0, 1.0)
    skin_a = np.clip(skin_alpha_u01, 0.0, 1.0)
    a = np.clip(acne_a * skin_a, 0.0, 1.0)[..., None]
    return high_f32 * (1.0 - a), spike_u8


def _high_layer_to_vis_u8(high_f32: np.ndarray) -> np.ndarray:
    """Map signed high-frequency residual to viewable uint8 (offset 128)."""
    return np.clip(high_f32 + 128.0, 0, 255).astype(np.uint8)


def process_face_skin_smooth_and_freq_enhance(
    rgb_u8: np.ndarray,
    *,
    skin_mask_u8: Optional[np.ndarray] = None,
    split_sigma: float = 12.0,
    high_gain: float = 1.0,
    avg_radius: int = 2,
    gauss_sigma: float = 0.7,
    fft_smooth_sigma_norm: float = 0.07,
    spatial_mix: Tuple[float, float, float] = (0.40, 0.22, 0.38),
    feather_radius: int = 14,
    skin_strength: float = 0.92,
    enable_high_acne_removal: bool = False,
    acne_spike_radius: int = 3,
    acne_spike_threshold: float = 7.0,
    acne_spike_rel_ratio: float = 1.45,
    acne_min_area: int = 2,
    acne_max_area: int = 350,
    acne_feather_radius: int = 4,
    acne_strength: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Frequency-separation skin pipeline.

    1. Decompose: low (Gaussian ``split_sigma``) + high (residual texture).
    2. Optional: high-layer spike detect + press-to-zero (inside skin).
    3. Smooth **low layer only** inside feathered skin mask.
    4. Recompose: ``low_processed + high_gain * high`` (high keeps pores/detail).
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be (H,W,3) uint8")

    if skin_mask_u8 is None:
        mask_skin, info_seg = segment_v4(rgb_u8)
    else:
        if skin_mask_u8.shape[:2] != rgb_u8.shape[:2]:
            raise ValueError("skin_mask_u8 shape must match rgb height/width")
        mask_skin = skin_mask_u8
        info_seg = {}
    alpha = feather_mask_u01(mask_skin, radius=feather_radius)
    alpha = np.clip(alpha * float(skin_strength), 0.0, 1.0)

    src_f32 = rgb_u8.astype(np.float32)
    low_f32, high_f32 = decompose_frequency_layers(src_f32, split_sigma=split_sigma)
    high_raw_f32 = high_f32
    acne_mask = np.zeros(mask_skin.shape[:2], dtype=np.uint8)

    if enable_high_acne_removal:
        high_f32, acne_mask = remove_acne_on_high_layer(
            high_f32,
            mask_skin,
            alpha,
            spike_radius=acne_spike_radius,
            spike_threshold=acne_spike_threshold,
            rel_ratio=acne_spike_rel_ratio,
            min_area=acne_min_area,
            max_area=acne_max_area,
            feather_radius=acne_feather_radius,
            strength=acne_strength,
        )

    low_processed = process_low_frequency_layer(
        low_f32,
        alpha,
        avg_radius=avg_radius,
        gauss_sigma=gauss_sigma,
        fft_smooth_sigma_norm=fft_smooth_sigma_norm,
        spatial_mix=spatial_mix,
    )

    final_rgb = recompose_frequency_layers(
        low_processed,
        high_f32,
        high_gain=high_gain,
    )

    aux: Dict[str, np.ndarray] = {
        "skin_mask": mask_skin,
        "acne_mask": acne_mask,
        "alpha_skin": (alpha * 255).astype(np.uint8),
        "low_layer": np.clip(low_f32, 0, 255).astype(np.uint8),
        "high_layer": _high_layer_to_vis_u8(high_f32),
        "high_layer_raw": _high_layer_to_vis_u8(high_raw_f32),
        "low_processed": np.clip(low_processed, 0, 255).astype(np.uint8),
        "segment_info": info_seg,
        # legacy keys for callers / intermediate saves
        "smooth_spatial_freq": np.clip(low_processed, 0, 255).astype(np.uint8),
        "after_smooth": final_rgb,
    }
    return final_rgb, aux


def _default_out_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, "output", "skin_smooth_freq")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skin: frequency separation (smooth low layer, preserve high texture)."
    )
    parser.add_argument("--input", type=str, help="Single image path")
    parser.add_argument("--input_folder", type=str, help="Batch input directory")
    parser.add_argument("--output", type=str, default="", help="Output directory (default: rey_code/output/skin_smooth_freq)")
    parser.add_argument(
        "--split_sigma",
        type=float,
        default=12.0,
        help="Layer split Gaussian sigma (px); larger = more detail in high layer",
    )
    parser.add_argument(
        "--high_gain",
        type=float,
        default=1.0,
        help="Scale high-frequency layer on recompose (1.0 = full texture)",
    )
    parser.add_argument("--avg_radius", type=int, default=2, help="Low-layer box-filter radius")
    parser.add_argument("--gauss_sigma", type=float, default=0.7, help="Extra low-layer Gaussian sigma")
    parser.add_argument("--fft_smooth_sigma", type=float, default=0.07, help="FFT low-pass on low layer")
    parser.add_argument("--feather", type=int, default=14, help="Skin mask feather radius (box blur)")
    parser.add_argument(
        "--high_acne",
        action="store_true",
        help="High-layer spike detect + press-to-zero",
    )
    parser.add_argument(
        "--acne_spike_thresh",
        type=float,
        default=7.0,
        help="|high| minus local mean threshold (larger = fewer spikes)",
    )
    parser.add_argument(
        "--acne_spike_ratio",
        type=float,
        default=1.45,
        help="Also flag if |high| > local_mean * ratio",
    )
    parser.add_argument("--acne_spike_radius", type=int, default=3, help="Local neighborhood radius (px)")
    parser.add_argument("--acne_strength", type=float, default=0.95, help="Press-to-zero strength 0..1")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate PNGs")
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Main output filename inside --output (default: <stem>_enhanced.png)",
    )

    args = parser.parse_args()
    out_dir = args.output.strip() or _default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    def run_one(path: str) -> None:
        rgb = read_rgb(path)
        final_rgb, aux = process_face_skin_smooth_and_freq_enhance(
            rgb,
            split_sigma=args.split_sigma,
            high_gain=args.high_gain,
            avg_radius=args.avg_radius,
            gauss_sigma=args.gauss_sigma,
            fft_smooth_sigma_norm=args.fft_smooth_sigma,
            feather_radius=args.feather,
            enable_high_acne_removal=args.high_acne,
            acne_spike_threshold=args.acne_spike_thresh,
            acne_spike_rel_ratio=args.acne_spike_ratio,
            acne_spike_radius=args.acne_spike_radius,
            acne_strength=args.acne_strength,
        )
        base = os.path.splitext(os.path.basename(path))[0]
        main_name = args.output_name.strip() or f"{base}_enhanced.png"
        save_image(final_rgb, os.path.join(out_dir, main_name))
        save_image(aux["skin_mask"], os.path.join(out_dir, f"{base}_skin_mask.png"))
        if args.save_intermediate:
            save_image(aux["low_layer"], os.path.join(out_dir, f"{base}_low.png"))
            save_image(aux["high_layer"], os.path.join(out_dir, f"{base}_high.png"))
            if args.high_acne and np.any(aux["acne_mask"] > 0):
                save_image(aux["high_layer_raw"], os.path.join(out_dir, f"{base}_high_raw.png"))
                save_image(aux["acne_mask"], os.path.join(out_dir, f"{base}_acne_mask.png"))
            save_image(aux["low_processed"], os.path.join(out_dir, f"{base}_low_smooth.png"))
            save_image(final_rgb, os.path.join(out_dir, f"{base}_recombined.png"))
            save_image(aux["alpha_skin"], os.path.join(out_dir, f"{base}_alpha.png"))
        print(f"OK: {path} -> {os.path.join(out_dir, main_name)}")

    if args.input:
        run_one(args.input)
        return
    if args.input_folder:
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = sorted(
            f for f in os.listdir(args.input_folder) if f.lower().endswith(exts)
        )
        if not files:
            print(f"No images in {args.input_folder}")
            return
        for f in files:
            run_one(os.path.join(args.input_folder, f))
        print(f"DONE. Output: {out_dir}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
