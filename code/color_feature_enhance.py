"""
color_feature_enhance.py
Member 4: Color Processing & Feature Enhancement Module (Lecture 7 + Lecture 4)

Implements:
- Skin-tone adjustment in RGB/HSV (warmth/whitening)
- Local intensity transforms for eyes/lips contrast (gamma + unsharp + saturation)
- Pseudo-color techniques (colormap on pseudo-luma)

This file is imported by `code/face_beauty_pipeline.py` as `color_feature_enhance`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _import_segmentation_with_stub():
    """
    Prefer using `segmentation.py` APIs (per assignment requirement).
    This repo's `segmentation.py` imports matplotlib; some environments have binary
    breakage (NumPy 2.x). We stub matplotlib modules to allow import when needed.
    """
    import sys
    import types

    # Always provide a lightweight `matplotlib.pyplot` shim so `segmentation.py`
    # can be imported without depending on compiled matplotlib wheels.
    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")

    if "matplotlib.pyplot" not in sys.modules:
        plt = types.ModuleType("matplotlib.pyplot")

        def imread(path: str):
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            # mimic matplotlib: float 0..1 for PNGs sometimes; we keep uint8
            return arr

        def imsave(path: str, array, cmap=None, vmin=None, vmax=None):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            a = np.asarray(array)
            if a.ndim == 2:
                Image.fromarray(a.astype(np.uint8)).save(path)
            else:
                if a.shape[2] == 4:
                    a = a[:, :, :3]
                Image.fromarray(a.astype(np.uint8)).save(path)

        plt.imread = imread  # type: ignore[attr-defined]
        plt.imsave = imsave  # type: ignore[attr-defined]
        sys.modules["matplotlib.pyplot"] = plt

    import segmentation as seg  # type: ignore
    return seg


# =========================
# Basic color conversions
# =========================

def rgb_to_hsv_u01(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H,W,3) -> HSV float32 in [0,1]."""
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be (H,W,3)")
    x = rgb_u8.astype(np.float32) / 255.0
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    s = np.zeros_like(cmax, dtype=np.float32)
    v = cmax.astype(np.float32)

    nz = delta > 1e-8
    s[nz] = (delta[nz] / (cmax[nz] + 1e-8)).astype(np.float32)

    r_is_max = (cmax == r) & nz
    g_is_max = (cmax == g) & nz
    b_is_max = (cmax == b) & nz
    h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / (delta[r_is_max] + 1e-8)) % 6.0
    h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / (delta[g_is_max] + 1e-8)) + 2.0
    h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / (delta[b_is_max] + 1e-8)) + 4.0
    h = (h / 6.0).astype(np.float32)

    return np.stack([h, s, v], axis=-1)


def hsv_u01_to_rgb_u8(hsv_u01: np.ndarray) -> np.ndarray:
    """Convert HSV float32 in [0,1] -> uint8 RGB."""
    if hsv_u01.ndim != 3 or hsv_u01.shape[2] != 3:
        raise ValueError("hsv_u01 must be (H,W,3)")
    h = (hsv_u01[:, :, 0].astype(np.float32) % 1.0) * 6.0
    s = np.clip(hsv_u01[:, :, 1].astype(np.float32), 0.0, 1.0)
    v = np.clip(hsv_u01[:, :, 2].astype(np.float32), 0.0, 1.0)

    c = v * s
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = v - c

    z = np.zeros_like(h, dtype=np.float32)
    rp = np.empty_like(h, dtype=np.float32)
    gp = np.empty_like(h, dtype=np.float32)
    bp = np.empty_like(h, dtype=np.float32)

    h0 = (0.0 <= h) & (h < 1.0)
    h1 = (1.0 <= h) & (h < 2.0)
    h2 = (2.0 <= h) & (h < 3.0)
    h3 = (3.0 <= h) & (h < 4.0)
    h4 = (4.0 <= h) & (h < 5.0)
    h5 = (5.0 <= h) & (h < 6.0)

    rp[h0], gp[h0], bp[h0] = c[h0], x[h0], z[h0]
    rp[h1], gp[h1], bp[h1] = x[h1], c[h1], z[h1]
    rp[h2], gp[h2], bp[h2] = z[h2], c[h2], x[h2]
    rp[h3], gp[h3], bp[h3] = z[h3], x[h3], c[h3]
    rp[h4], gp[h4], bp[h4] = x[h4], z[h4], c[h4]
    rp[h5], gp[h5], bp[h5] = c[h5], z[h5], x[h5]

    r = (rp + m) * 255.0
    g = (gp + m) * 255.0
    b = (bp + m) * 255.0
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_to_ycbcr_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """YCbCr conversion compatible with Member1."""
    rgb_f = rgb_u8.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128.0
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


# =========================
# Small image helpers
# =========================

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


def _box_blur_rgb_u8(rgb_u8: np.ndarray, radius: int) -> np.ndarray:
    """
    Box blur for RGB using integral image, fully vectorized over channels.

    This keeps the exact same box-blur math as the per-channel version, but
    removes the Python-level loop over RGB channels.
    """
    if radius <= 0:
        return rgb_u8.astype(np.uint8, copy=False)
    r = int(radius)
    x = rgb_u8.astype(np.float32)
    h, w, ch = x.shape
    if ch != 3:
        raise ValueError("rgb_u8 must be (H,W,3)")

    padded = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge")
    integ = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1, ch), dtype=np.float32)
    integ[1:, 1:, :] = padded.cumsum(axis=0).cumsum(axis=1)

    y0 = np.arange(0, h, dtype=np.int32)
    y1 = y0 + (2 * r + 1)
    x0 = np.arange(0, w, dtype=np.int32)
    x1 = x0 + (2 * r + 1)

    a = integ[y0[:, None], x0[None, :], :]
    b = integ[y0[:, None], x1[None, :], :]
    c = integ[y1[:, None], x0[None, :], :]
    d = integ[y1[:, None], x1[None, :], :]
    area = float((2 * r + 1) ** 2)
    out = (d - b - c + a) / area
    return np.clip(out, 0, 255).astype(np.uint8)


def _feather_mask_u01(mask_u8: np.ndarray, radius: int = 12) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if radius <= 0:
        return m
    return np.clip(_box_blur_gray_f32(m, radius=max(1, int(radius))), 0.0, 1.0)


def _blend(original_u8: np.ndarray, processed_u8: np.ndarray, alpha_u01: np.ndarray) -> np.ndarray:
    a = np.clip(alpha_u01[..., None], 0.0, 1.0).astype(np.float32)
    o = original_u8.astype(np.float32)
    p = processed_u8.astype(np.float32)
    return np.clip(o * (1.0 - a) + p * a, 0, 255).astype(np.uint8)


def enhance_contrast_rgb_u8(rgb_u8: np.ndarray, low_p: float = 1.0, high_p: float = 99.0) -> np.ndarray:
    """Per-channel percentile stretch (simple, robust contrast enhancement)."""
    x = rgb_u8.astype(np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("rgb_u8 must be (H,W,3)")

    # Percentiles per channel (vectorized over RGB channels)
    lo = np.percentile(x, low_p, axis=(0, 1))  # (3,)
    hi = np.percentile(x, high_p, axis=(0, 1))  # (3,)
    den = hi - lo
    ok = den > 1e-6  # (3,)

    # Compute stretched values for channels where ok=true
    factor = np.ones_like(den, dtype=np.float32)
    factor[ok] = 255.0 / den[ok]
    out_stretch = (x - lo[None, None, :]) * factor[None, None, :]

    # For channels where hi ~ lo, keep original values
    out = np.where(ok.reshape(1, 1, 3), out_stretch, x)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# Skin mask (adaptive YCbCr)
# =========================

def ycbcr_skin_mask_u8_adaptive(
    rgb_u8: np.ndarray,
    *,
    base_cb_range: Tuple[int, int] = (77, 127),
    base_cr_range: Tuple[int, int] = (133, 173),
    widen: int = 6,
    min_pixels: int = 200,
) -> np.ndarray:
    """
    Adaptive skin mask:
    - Start from classic fixed Cb/Cr ranges (Member1-like).
    - If enough candidates, adapt thresholds around candidate mean/std (widened slightly).
    Returns uint8 mask in {0,255}.
    """
    ycbcr = rgb_to_ycbcr_u8(rgb_u8)
    cb = ycbcr[:, :, 1].astype(np.int32)
    cr = ycbcr[:, :, 2].astype(np.int32)

    cb0, cb1 = base_cb_range
    cr0, cr1 = base_cr_range
    cand = (cb >= cb0) & (cb <= cb1) & (cr >= cr0) & (cr <= cr1)

    if int(cand.sum()) < int(min_pixels):
        return (cand.astype(np.uint8) * 255)

    cb_c = cb[cand].astype(np.float32)
    cr_c = cr[cand].astype(np.float32)
    cb_mu, cb_sd = float(cb_c.mean()), float(cb_c.std() + 1e-6)
    cr_mu, cr_sd = float(cr_c.mean()), float(cr_c.std() + 1e-6)

    cb_lo = int(np.clip(cb_mu - 2.2 * cb_sd - widen, 0, 255))
    cb_hi = int(np.clip(cb_mu + 2.2 * cb_sd + widen, 0, 255))
    cr_lo = int(np.clip(cr_mu - 2.2 * cr_sd - widen, 0, 255))
    cr_hi = int(np.clip(cr_mu + 2.2 * cr_sd + widen, 0, 255))

    mask = (cb >= cb_lo) & (cb <= cb_hi) & (cr >= cr_lo) & (cr <= cr_hi)
    return (mask.astype(np.uint8) * 255)


# =========================
# Skin tone adjustment
# =========================

@dataclass
class SkinToneParams:
    # Whitening/brightening on skin
    whiten_strength: float = 0.85  # scales V lift (smooth curve, not used for blur)
    v_gain: float = 1.06           # HSV V multiplicative target toward this gain
    s_gain: float = 0.93           # HSV S multiplicative target
    hue_shift: float = -2.0        # degrees
    # RGB warmth (small auxiliary blend, kept subtle to avoid color clipping)
    rgb_gain: Tuple[float, float, float] = (1.03, 1.01, 0.985)
    rgb_bias: Tuple[float, float, float] = (2.0, 1.0, 0.0)
    rgb_warmth_mix: float = 0.28   # 0..1 portion of warmth mixed under skin alpha
    # Legacy:磨皮 off by default (no spatial blur on skin)
    smooth_strength: float = 0.0
    smooth_radius: int = 0
    detail_amount: float = 0.0
    post_smooth_strength: float = 0.0
    # Mask feathering (spatially continuous alpha)
    feather_radius: int = 14


def adjust_skin_tone_rgb(rgb_u8: np.ndarray, skin_mask_u8: np.ndarray, params: SkinToneParams) -> np.ndarray:
    x = rgb_u8.astype(np.float32)
    gain = np.array(params.rgb_gain, dtype=np.float32)
    bias = np.array(params.rgb_bias, dtype=np.float32)
    y = x * gain[None, None, :] + bias[None, None, :]
    y = np.clip(y, 0, 255).astype(np.uint8)

    alpha = _feather_mask_u01(skin_mask_u8, radius=params.feather_radius)
    return _blend(rgb_u8, y, alpha)


def adjust_skin_tone_hsv(rgb_u8: np.ndarray, skin_mask_u8: np.ndarray, params: SkinToneParams) -> np.ndarray:
    hsv = rgb_to_hsv_u01(rgb_u8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    m = skin_mask_u8 > 127
    if np.any(m):
        # hue shift in degrees -> [0,1]
        dh = float(params.hue_shift) / 360.0
        h2 = h.copy()
        s2 = s.copy()
        v2 = v.copy()
        h2[m] = (h2[m] + dh) % 1.0
        s2[m] = np.clip(s2[m] * float(params.s_gain), 0.0, 1.0)
        v2[m] = np.clip(v2[m] * float(params.v_gain), 0.0, 1.0)
        rgb2 = hsv_u01_to_rgb_u8(np.stack([h2, s2, v2], axis=-1))
    else:
        rgb2 = rgb_u8

    alpha = _feather_mask_u01(skin_mask_u8, radius=params.feather_radius)
    return _blend(rgb_u8, rgb2, alpha)


def skin_tone_adjust(
    rgb_u8: np.ndarray,
    skin_mask_u8: np.ndarray,
    params: Optional[SkinToneParams] = None,
) -> np.ndarray:
    """Combined RGB warmth + HSV whitening; blended in skin region."""
    p = params or SkinToneParams()
    a = adjust_skin_tone_rgb(rgb_u8, skin_mask_u8, p)
    b = adjust_skin_tone_hsv(a, skin_mask_u8, p)
    # extra controlled whitening blend
    alpha = _feather_mask_u01(skin_mask_u8, radius=p.feather_radius) * float(p.whiten_strength)
    return _blend(rgb_u8, b, np.clip(alpha, 0.0, 1.0))


def smooth_skin_color_enhance(
    rgb_u8: np.ndarray,
    skin_mask_u8: np.ndarray,
    params: SkinToneParams,
    skin_strength: float,
) -> np.ndarray:
    """
    Spatially continuous skin color enhancement (no磨皮): feathered alpha × smooth
    HSV + mild RGB warmth, with tanh curves to limit per-channel exaggeration and
    highlight suppression to preserve specular detail.
    Fully vectorized over the image grid.
    """
    ss = float(max(0.0, skin_strength))
    if ss < 1e-6 or not np.any(skin_mask_u8 > 127):
        return rgb_u8

    p = params
    a = _feather_mask_u01(skin_mask_u8, radius=int(p.feather_radius)).astype(np.float32)
    a = np.clip(a * ss, 0.0, 1.0)
    a = _suppress_highlights_alpha(rgb_u8, a, thr=0.86)

    hsv = rgb_to_hsv_u01(rgb_u8)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    dh_deg = float(p.hue_shift) / 360.0
    dh = a * np.tanh(dh_deg * 18.0) / 18.0
    h2 = (h + dh) % 1.0

    g_s = float(p.s_gain)
    s_delta = a * np.tanh((s * (g_s - 1.0)) * 3.0) * 0.22
    s2 = np.clip(s + s_delta, 0.0, 1.0)

    g_v = float(p.v_gain)
    v_lin = a * np.tanh((v * (g_v - 1.0)) * 2.8) * 0.2
    wh = float(np.clip(p.whiten_strength, 0.0, 1.25))
    hi_gate = np.clip((v - 0.74) / 0.26, 0.0, 1.0)
    v_w = a * wh * 0.07 * (1.0 - v) * (1.0 - 0.9 * hi_gate)
    v_w = np.minimum(v_w, 0.085)
    v2 = np.clip(v + v_lin + v_w, 0.0, 0.998)

    rgb_hsv = hsv_u01_to_rgb_u8(np.stack([h2, s2, v2], axis=-1)).astype(np.float32)
    x = rgb_u8.astype(np.float32)
    gain = np.array(p.rgb_gain, dtype=np.float32)
    bias = np.array(p.rgb_bias, dtype=np.float32)
    warm = x * gain[None, None, :] + bias[None, None, :]
    wm = float(np.clip(p.rgb_warmth_mix, 0.0, 1.0))
    aw = (a[..., None] ** 0.92) * wm
    mixed = rgb_hsv * (1.0 - aw) + warm * aw
    out = x * (1.0 - a[..., None]) + mixed * a[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def whiten_and_smooth_skin(
    rgb_u8: np.ndarray,
    skin_mask_u8: np.ndarray,
    *,
    whiten_strength: float = 0.85,
    smooth_strength: float = 0.55,
    smooth_radius: int = 2,
    detail_amount: float = 0.35,
    feather_radius: int = 14,
) -> np.ndarray:
    """
    Whitening + smoothing only on skin:
    - smoothing: box blur on RGB
    - whitening: lift V a bit and reduce S a bit (HSV)
    """
    if not np.any(skin_mask_u8 > 127):
        return rgb_u8

    # smoothing branch (mild), blended by alpha to avoid global blur
    smooth = _box_blur_rgb_u8(rgb_u8, radius=max(0, int(smooth_radius)))
    alpha_s = _feather_mask_u01(skin_mask_u8, radius=feather_radius)
    alpha_s = np.clip(alpha_s * float(np.clip(smooth_strength, 0.0, 1.0)), 0.0, 1.0)
    base = _blend(rgb_u8, smooth, alpha_s)

    # whitening branch in HSV (adaptive: avoid over-whitening bright skin)
    hsv = rgb_to_hsv_u01(base)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = skin_mask_u8 > 127
    v2 = v.copy()
    s2 = s.copy()
    if np.any(m):
        v_mean = float(np.clip(v[m].mean(), 0.0, 1.0))
        # target mean brightness increases with whiten_strength but caps to avoid "plastic"
        target = float(np.clip(v_mean + 0.10 * float(np.clip(whiten_strength, 0.0, 1.0)), 0.0, 0.88))
        # push v toward target with a soft curve
        v2[m] = np.clip(v2[m] + (target - v2[m]) * (0.85 * float(np.clip(whiten_strength, 0.0, 1.0))), 0.0, 1.0)
        s2[m] = np.clip(s2[m] * (1.0 - 0.10 * float(np.clip(whiten_strength, 0.0, 1.0))), 0.0, 1.0)
    white = hsv_u01_to_rgb_u8(np.stack([h, s2, v2], axis=-1))

    # apply highlight suppression to avoid shiny over-processing
    alpha_w = _feather_mask_u01(skin_mask_u8, radius=feather_radius)
    alpha_w = _suppress_highlights_alpha(base, alpha_w, thr=0.90)
    alpha_w = np.clip(alpha_w * float(np.clip(whiten_strength, 0.0, 1.0)), 0.0, 1.0)
    out = _blend(base, white, alpha_w)

    # detail add-back (high-frequency reinjection) to reduce "blur"
    da = float(np.clip(detail_amount, 0.0, 1.0))
    if da > 1e-6 and np.any(m):
        blur2 = _box_blur_rgb_u8(out, radius=1).astype(np.float32)
        out_f = out.astype(np.float32)
        detail = out_f - blur2
        sharpen = np.clip(out_f + da * detail, 0, 255).astype(np.uint8)
        alpha_d = _feather_mask_u01(skin_mask_u8, radius=max(6, feather_radius // 2))
        alpha_d = _suppress_highlights_alpha(out, alpha_d, thr=0.93)
        alpha_d = np.clip(alpha_d * (0.65 * float(np.clip(smooth_strength, 0.0, 1.0))), 0.0, 1.0)
        out = _blend(out, sharpen, alpha_d)

    return out


# =========================
# Face region estimation (from skin mask)
# =========================

def _connected_components_bboxes(binary_mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """
    Simple 4-connected component labeling (no scipy).
    Returns list of (area, y0, x0, y1, x1), y1/x1 are exclusive.
    """
    m = (binary_mask > 0).astype(np.uint8)
    h, w = m.shape
    visited = np.zeros_like(m, dtype=np.uint8)
    out: List[Tuple[int, int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x] != 0:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            area = 0
            y0 = y1 = y
            x0 = x1 = x
            while stack:
                cy, cx = stack.pop()
                area += 1
                y0 = min(y0, cy)
                x0 = min(x0, cx)
                y1 = max(y1, cy)
                x1 = max(x1, cx)
                if cy > 0 and m[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = 1
                    stack.append((cy - 1, cx))
                if cy + 1 < h and m[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = 1
                    stack.append((cy + 1, cx))
                if cx > 0 and m[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = 1
                    stack.append((cy, cx - 1))
                if cx + 1 < w and m[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = 1
                    stack.append((cy, cx + 1))
            out.append((area, y0, x0, y1 + 1, x1 + 1))

    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _skin_bboxes_fast(
    skin_mask_u8: np.ndarray,
    *,
    min_area: int,
    min_col_coverage: float = 0.01,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Vectorized multi-face bbox extraction from a skin mask.

    Idea: faces appear as separate "islands" in the x-projection of the skin mask.
    We find contiguous x segments where skin coverage is non-trivial, then for each
    segment compute y-range. This avoids slow Python connected-component BFS.

    Returns list of (area, y0, x0, y1, x1) sorted by area desc.
    """
    m = (skin_mask_u8 > 127)
    h, w = m.shape
    if not np.any(m):
        return []

    col_cnt = m.sum(axis=0).astype(np.int32)
    # column is "active" if it has enough skin pixels
    thr = max(1, int(min_col_coverage * h))
    active = col_cnt >= thr
    if not np.any(active):
        return []

    a = active.astype(np.uint8)
    d = np.diff(np.pad(a, (1, 1), mode="constant"))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if starts.size == 0 or ends.size == 0:
        return []

    bboxes: List[Tuple[int, int, int, int, int]] = []
    for x0, x1 in zip(starts.tolist(), ends.tolist()):
        if x1 <= x0:
            continue
        roi = m[:, x0:x1]
        area = int(roi.sum())
        if area < int(min_area):
            continue
        rows = np.flatnonzero(roi.any(axis=1))
        if rows.size == 0:
            continue
        y0 = int(rows[0])
        y1 = int(rows[-1] + 1)
        bboxes.append((area, y0, x0, y1, x1))

    bboxes.sort(key=lambda t: t[0], reverse=True)
    return bboxes


@dataclass
class FaceRegions:
    face_bbox: Tuple[int, int, int, int]  # (y0, x0, y1, x1) exclusive
    eye_bbox: Tuple[int, int, int, int]
    lip_bbox: Tuple[int, int, int, int]


def estimate_face_regions_from_skin_mask(skin_mask_u8: np.ndarray) -> FaceRegions:
    """
    Estimate face / eye / lip boxes using simple anthropometric ratios within the
    largest skin connected component.
    """
    h, w = skin_mask_u8.shape[:2]
    comps = _connected_components_bboxes((skin_mask_u8 > 127).astype(np.uint8))
    if not comps:
        # fallback: whole image
        face = (0, 0, h, w)
    else:
        _, y0, x0, y1, x1 = comps[0]
        face = (y0, x0, y1, x1)

    fy0, fx0, fy1, fx1 = face
    fh = max(1, fy1 - fy0)
    fw = max(1, fx1 - fx0)

    # Eye region: upper-middle band
    ey0 = int(np.clip(fy0 + 0.22 * fh, 0, h))
    ey1 = int(np.clip(fy0 + 0.48 * fh, 0, h))
    ex0 = int(np.clip(fx0 + 0.12 * fw, 0, w))
    ex1 = int(np.clip(fx1 - 0.12 * fw, 0, w))
    eye = (ey0, ex0, max(ey0 + 1, ey1), max(ex0 + 1, ex1))

    # Lip region: lower-middle band
    ly0 = int(np.clip(fy0 + 0.62 * fh, 0, h))
    ly1 = int(np.clip(fy0 + 0.86 * fh, 0, h))
    lx0 = int(np.clip(fx0 + 0.18 * fw, 0, w))
    lx1 = int(np.clip(fx1 - 0.18 * fw, 0, w))
    lip = (ly0, lx0, max(ly0 + 1, ly1), max(lx0 + 1, lx1))

    return FaceRegions(face_bbox=face, eye_bbox=eye, lip_bbox=lip)


def estimate_eye_lip_masks_u8(
    rgb_u8: np.ndarray,
    skin_mask_u8: np.ndarray,
    regions: Optional[FaceRegions] = None,
    *,
    feather_radius: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-face + multi-scale geometry-based masks (robust fallback).

    Per connected component (face candidate) from `skin_mask_u8`, estimate face bbox,
    then place:
    - two eye rectangles (left/right) in the upper-mid band
    - one lip rectangle in the lower band

    This avoids relying on fragile color heuristics and works across scales/multi-face.
    """
    h, w = rgb_u8.shape[:2]

    def keep_largest(mask01: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return np.zeros_like(mask01, dtype=np.uint8)
        comps = _connected_components_bboxes(mask01.astype(np.uint8))
        if not comps:
            return np.zeros_like(mask01, dtype=np.uint8)
        outm = np.zeros_like(mask01, dtype=np.uint8)
        for area, y0, x0, y1, x1 in comps[:k]:
            outm[y0:y1, x0:x1] |= mask01[y0:y1, x0:x1].astype(np.uint8)
        return outm

    # Build per-face bboxes from skin connected components (multi-face support)
    skin_bin = (skin_mask_u8 > 127).astype(np.uint8)
    comps = _connected_components_bboxes(skin_bin)
    # Filter tiny components (noise) relative to image size
    min_area = max(200, int(0.0025 * h * w))
    face_boxes = [(a, y0, x0, y1, x1) for (a, y0, x0, y1, x1) in comps if a >= min_area]
    if not face_boxes:
        # fallback to single region estimation if no components
        reg = regions or estimate_face_regions_from_skin_mask(skin_mask_u8)
        fy0, fx0, fy1, fx1 = reg.face_bbox
        face_boxes = [(int((fy1 - fy0) * (fx1 - fx0)), fy0, fx0, fy1, fx1)]

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    lip_mask = np.zeros((h, w), dtype=np.uint8)

    for _, fy0, fx0, fy1, fx1 in face_boxes:
        fh = max(1, fy1 - fy0)
        fw = max(1, fx1 - fx0)

        y0, y1 = fy0, fy1
        x0, x1 = fx0, fx1

        # Eye band and lip band
        band_eye_y0 = int(np.clip(y0 + 0.22 * fh, 0, h))
        band_eye_y1 = int(np.clip(y0 + 0.48 * fh, 0, h))
        band_lip_y0 = int(np.clip(y0 + 0.62 * fh, 0, h))
        band_lip_y1 = int(np.clip(y0 + 0.86 * fh, 0, h))

        # Two eye rectangles (left/right) inside eye band
        eye_w = int(max(2, 0.22 * fw))
        eye_h = int(max(2, (band_eye_y1 - band_eye_y0)))
        eye_gap = int(max(2, 0.06 * fw))
        eye_center_y0 = band_eye_y0
        eye_center_y1 = band_eye_y0 + eye_h
        midx = (x0 + x1) // 2
        left_x1 = int(np.clip(midx - eye_gap // 2, 0, w))
        left_x0 = int(np.clip(left_x1 - eye_w, 0, w))
        right_x0 = int(np.clip(midx + eye_gap // 2, 0, w))
        right_x1 = int(np.clip(right_x0 + eye_w, 0, w))
        eye_mask[eye_center_y0:eye_center_y1, left_x0:left_x1] = 255
        eye_mask[eye_center_y0:eye_center_y1, right_x0:right_x1] = 255

        # Lip rectangle centered horizontally in lip band
        lip_y0 = band_lip_y0
        lip_y1 = band_lip_y1
        lip_w = int(max(2, 0.42 * fw))
        lip_x0 = int(np.clip(midx - lip_w // 2, 0, w))
        lip_x1 = int(np.clip(lip_x0 + lip_w, 0, w))
        lip_mask[lip_y0:lip_y1, lip_x0:lip_x1] = 255

    # Feather output masks (scale-aware)
    fr = int(max(4, feather_radius))
    eye_f = (_feather_mask_u01(eye_mask, radius=fr) * 255.0).astype(np.uint8)
    lip_f = (_feather_mask_u01(lip_mask, radius=fr) * 255.0).astype(np.uint8)
    return eye_f, lip_f


# =========================
# Eyes / lips enhancement (local transforms)
# =========================

@dataclass
class FeatureEnhanceParams:
    # Eyes
    eye_v_gamma: float = 0.70
    eye_unsharp_radius: int = 5
    eye_unsharp_amount: float = 1.65
    eye_s_gain: float = 0.85
    # Extra smooth contrast curve in eye region (applied in soft path)
    eye_contrast_strength: float = 0.92  # 0..~1, blended by smooth eye weights
    eye_contrast_gain: float = 3.6       # tanh gain, higher = stronger midtone contrast
    # Extra local contrast (unsharp-like) on V inside eye region, blended by smooth weights
    eye_local_contrast_strength: float = 1.05  # 0..~1.5
    eye_local_contrast_radius: int = 4        # box blur radius on V (smoother)
    eye_local_contrast_limit: float = 0.12    # max abs delta on V (before blend)
    # Lips
    lip_v_gamma: float = 0.95
    lip_unsharp_radius: int = 4
    lip_unsharp_amount: float = 1.25
    lip_s_gain: float = 1.55
    lip_hue_shift: float = 0.0  # degrees
    # Mask feather
    feather_radius: int = 8
    # Weight-map detection kernels (two scales per group)
    eye_kernel_r_small: int = 18
    eye_kernel_r_mid: int = 26
    eye_kernel_r_large: int = 34
    lip_kernel_r_small: int = 16
    lip_kernel_r_mid: int = 23
    lip_kernel_r_large: int = 30
    # Weight-guided smoothing radii
    eye_smooth_radius: int = 2
    lip_smooth_radius: int = 1
    # Expand + smooth weight maps so enhancement is spatially continuous (mask + neighborhood)
    weight_smooth_blur_r: int = 0  # 0 = auto from image size
    weight_smooth_iterations: int = 4
    # Contrast remap on smooth weights: boost peaks (capped), suppress dark tails
    weight_contrast_p_lo: float = 7.0
    weight_contrast_p_hi: float = 96.0
    weight_shoulder: float = 0.10
    weight_gamma: float = 0.62
    weight_peak_boost: float = 1.22
    weight_cap: float = 0.97


def _auto_weight_blur_r(h: int, w: int) -> int:
    return int(np.clip(min(h, w) // 35, 8, 28))


def smooth_feature_weights_for_enhancement(
    weight_u01: np.ndarray,
    params: Optional[FeatureEnhanceParams] = None,
    *,
    image_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Turn a rough eye/lip response into a smooth weight field: large-support blur
    (approx. Gaussian via repeated box) so alpha varies continuously in space and
    reaches neighboring pixels (mask + vicinity).
    """
    p = params or FeatureEnhanceParams()
    u = np.clip(weight_u01.astype(np.float32), 0.0, 1.0)
    if image_hw is None:
        image_hw = u.shape[:2]
    hh, ww = int(image_hw[0]), int(image_hw[1])
    r = int(p.weight_smooth_blur_r)
    if r <= 0:
        r = _auto_weight_blur_r(hh, ww)
    n = int(max(1, p.weight_smooth_iterations))
    for _ in range(n):
        u = _box_blur_gray_f32(u, radius=r)
    u = np.clip(u, 0.0, 1.0)
    mx = float(u.max())
    if mx > 1e-6:
        u = (u / mx).astype(np.float32)
    return u


def contrast_enhance_feature_weights(weight_u01: np.ndarray, params: Optional[FeatureEnhanceParams] = None) -> np.ndarray:
    """
    Vectorized S-curve style remap: stretch dynamic range, apply shoulder to keep
    near-black low, gamma < 1 to lift mid/high response, peak_boost + hard cap.
    """
    p = params or FeatureEnhanceParams()
    w = np.clip(weight_u01.astype(np.float32), 0.0, 1.0)
    lo = float(np.percentile(w, float(p.weight_contrast_p_lo)))
    hi = float(np.percentile(w, float(p.weight_contrast_p_hi)))
    t = (w - lo) / (hi - lo + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    sh = float(np.clip(p.weight_shoulder, 0.0, 0.35))
    t = np.clip((t - sh) / (1.0 - sh + 1e-6), 0.0, 1.0)
    g = float(np.clip(p.weight_gamma, 0.35, 1.2))
    t = np.power(t + 1e-8, g)
    t = t * t * (3.0 - 2.0 * t)
    t = np.clip(t * float(p.weight_peak_boost), 0.0, float(np.clip(p.weight_cap, 0.5, 1.0)))
    return t.astype(np.float32)


def _unsharp_rgb_u8(rgb_u8: np.ndarray, radius: int, amount: float) -> np.ndarray:
    if radius <= 0 or amount <= 0:
        return rgb_u8
    blur = _box_blur_rgb_u8(rgb_u8, radius=int(radius))
    x = rgb_u8.astype(np.float32)
    b = blur.astype(np.float32)
    sharp = x + float(amount) * (x - b)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _gamma_on_v(rgb_u8: np.ndarray, gamma: float) -> np.ndarray:
    hsv = rgb_to_hsv_u01(rgb_u8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g = float(max(gamma, 1e-4))
    v2 = np.clip(v, 0.0, 1.0) ** g
    return hsv_u01_to_rgb_u8(np.stack([h, s, v2], axis=-1))


def enhance_eyes_and_lips(
    rgb_u8: np.ndarray,
    eye_mask_u8: np.ndarray,
    lip_mask_u8: np.ndarray,
    params: Optional[FeatureEnhanceParams] = None,
    *,
    eye_strength: float = 1.0,
    lip_strength: float = 1.0,
) -> np.ndarray:
    p = params or FeatureEnhanceParams()
    out = rgb_u8

    es = float(max(0.0, eye_strength))
    ls = float(max(0.0, lip_strength))

    # Eyes: brighten V via gamma (<1), sharpen, slightly reduce saturation
    if np.any(eye_mask_u8 > 5) and es > 1e-6:
        eye_gamma = 1.0 + (float(p.eye_v_gamma) - 1.0) * es
        eye_unsharp_amt = 1.0 + (float(p.eye_unsharp_amount) - 1.0) * es
        eye_s_gain = 1.0 + (float(p.eye_s_gain) - 1.0) * es
        eye_branch = _gamma_on_v(out, gamma=float(eye_gamma))
        eye_branch = _unsharp_rgb_u8(eye_branch, radius=int(p.eye_unsharp_radius), amount=float(eye_unsharp_amt))
        hsv = rgb_to_hsv_u01(eye_branch)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        m = eye_mask_u8 > 5
        s2 = s.copy()
        s2[m] = np.clip(s2[m] * float(eye_s_gain), 0.0, 1.0)
        eye_branch = hsv_u01_to_rgb_u8(np.stack([h, s2, v], axis=-1))
        alpha_eye = _feather_mask_u01(eye_mask_u8, radius=p.feather_radius)
        alpha_eye = np.clip(alpha_eye * min(1.0, es), 0.0, 1.0)
        out = _blend(out, eye_branch, alpha_eye)

    # Lips: moderate gamma, sharpen, increase saturation, optional hue shift
    if np.any(lip_mask_u8 > 5) and ls > 1e-6:
        lip_gamma = 1.0 + (float(p.lip_v_gamma) - 1.0) * ls
        lip_unsharp_amt = 1.0 + (float(p.lip_unsharp_amount) - 1.0) * ls
        lip_s_gain = 1.0 + (float(p.lip_s_gain) - 1.0) * ls
        lip_branch = _gamma_on_v(out, gamma=float(lip_gamma))
        lip_branch = _unsharp_rgb_u8(lip_branch, radius=int(p.lip_unsharp_radius), amount=float(lip_unsharp_amt))
        hsv = rgb_to_hsv_u01(lip_branch)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        m = lip_mask_u8 > 5
        h2 = h.copy()
        s2 = s.copy()
        dh = float(p.lip_hue_shift) / 360.0
        h2[m] = (h2[m] + dh) % 1.0
        s2[m] = np.clip(s2[m] * float(lip_s_gain), 0.0, 1.0)
        lip_branch = hsv_u01_to_rgb_u8(np.stack([h2, s2, v], axis=-1))
        alpha_lip = _feather_mask_u01(lip_mask_u8, radius=p.feather_radius)
        alpha_lip = np.clip(alpha_lip * min(1.0, ls), 0.0, 1.0)
        out = _blend(out, lip_branch, alpha_lip)

    return out


# =========================
# Pseudo-color (Lecture 4)
# =========================

def apply_pseudocolor(gray_u8: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """
    Map a grayscale image to pseudo-color using a lightweight built-in colormap.
    Returns uint8 RGB (H,W,3).
    """
    if gray_u8.ndim == 3:
        # accept RGB input by converting to luma
        x = gray_u8.astype(np.float32)
        gray = (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]).astype(np.float32)
        gray_u01 = np.clip(gray / 255.0, 0.0, 1.0)
    else:
        gray_u01 = np.clip(gray_u8.astype(np.float32) / 255.0, 0.0, 1.0)

    name = (cmap or "jet").lower().strip()
    x = gray_u01.astype(np.float32)

    def jet(t: np.ndarray) -> np.ndarray:
        # classic "jet"-like piecewise ramps
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    def hot(t: np.ndarray) -> np.ndarray:
        r = np.clip(3.0 * t, 0.0, 1.0)
        g = np.clip(3.0 * t - 1.0, 0.0, 1.0)
        b = np.clip(3.0 * t - 2.0, 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    def cool(t: np.ndarray) -> np.ndarray:
        r = t
        g = 1.0 - t
        b = np.ones_like(t, dtype=np.float32)
        return np.stack([r, g, b], axis=-1)

    def gray(t: np.ndarray) -> np.ndarray:
        return np.stack([t, t, t], axis=-1)

    if name in {"jet", "j"}:
        rgb_u01 = jet(x)
    elif name in {"hot"}:
        rgb_u01 = hot(x)
    elif name in {"cool"}:
        rgb_u01 = cool(x)
    elif name in {"gray", "grey"}:
        rgb_u01 = gray(x)
    else:
        rgb_u01 = jet(x)

    return np.clip(rgb_u01 * 255.0, 0, 255).astype(np.uint8)


# =========================
# Alpha shaping utilities
# =========================

def _boost_soft_alpha(alpha_u01: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Make soft masks stronger without hard edges."""
    a = np.clip(alpha_u01.astype(np.float32), 0.0, 1.0)
    p = float(max(power, 1e-4))
    return np.clip(a ** p, 0.0, 1.0)


def _suppress_highlights_alpha(rgb_u8: np.ndarray, alpha_u01: np.ndarray, thr: float = 0.90) -> np.ndarray:
    """
    Reduce effect in saturated highlights to avoid 'plastic' shine:
    if luma is very high, scale alpha down.
    """
    x = rgb_u8.astype(np.float32) / 255.0
    luma = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    a = np.clip(alpha_u01.astype(np.float32), 0.0, 1.0)
    t = float(np.clip(thr, 0.0, 1.0))
    scale = np.ones_like(a, dtype=np.float32)
    hi = luma > t
    if np.any(hi):
        # linear falloff from t..1.0
        scale[hi] = np.clip(1.0 - (luma[hi] - t) / max(1e-6, (1.0 - t)), 0.25, 1.0)
    return np.clip(a * scale, 0.0, 1.0)


# =========================
# Main API for integration (Member5)
# =========================

@dataclass
class EnhanceOutputs:
    enhanced_rgb_u8: np.ndarray
    skin_mask_u8: np.ndarray
    eye_mask_u8: np.ndarray
    lip_mask_u8: np.ndarray
    pseudo_luma_u8: np.ndarray
    pseudocolor_rgb_u8: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _max_filter_gray_f32(gray: np.ndarray, radius: int) -> np.ndarray:
    """Fast max filter using sliding windows (small radius only)."""
    r = int(radius)
    if r <= 0:
        return gray.astype(np.float32, copy=False)
    from numpy.lib.stride_tricks import sliding_window_view

    g = gray.astype(np.float32, copy=False)
    p = np.pad(g, ((r, r), (r, r)), mode="edge")
    win = sliding_window_view(p, (2 * r + 1, 2 * r + 1))
    return win.max(axis=(-1, -2)).astype(np.float32)


def _min_filter_gray_f32(gray: np.ndarray, radius: int) -> np.ndarray:
    """Fast min filter using sliding windows (small radius only)."""
    r = int(radius)
    if r <= 0:
        return gray.astype(np.float32, copy=False)
    from numpy.lib.stride_tricks import sliding_window_view

    g = gray.astype(np.float32, copy=False)
    p = np.pad(g, ((r, r), (r, r)), mode="edge")
    win = sliding_window_view(p, (2 * r + 1, 2 * r + 1))
    return win.min(axis=(-1, -2)).astype(np.float32)


def _soft_eye_lip_weights(
    rgb_u8: np.ndarray,
    skin_mask_u8: np.ndarray,
    *,
    multi_face_min_area_ratio: float = 0.0025,
    eye_kernel_r_small: int = 18,
    eye_kernel_r_mid: int = 26,
    eye_kernel_r_large: int = 34,
    lip_kernel_r_small: int = 16,
    lip_kernel_r_mid: int = 23,
    lip_kernel_r_large: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build *soft* weights (0..1) for eyes and lips WITHOUT producing binary masks.
    Works per connected skin component (multi-face) using face bboxes.
    """
    h, w = rgb_u8.shape[:2]
    skin_bin = (skin_mask_u8 > 127).astype(np.uint8)
    min_area = max(200, int(float(multi_face_min_area_ratio) * h * w))
    # Fast vectorized bbox extraction (preferred)
    face_boxes = _skin_bboxes_fast(skin_mask_u8, min_area=min_area)
    # Fallback: full connected components (slower but more general)
    if not face_boxes:
        comps = _connected_components_bboxes(skin_bin)
        face_boxes = [(a, y0, x0, y1, x1) for (a, y0, x0, y1, x1) in comps if a >= min_area]
    if not face_boxes:
        face_boxes = [(h * w, 0, 0, h, w)]

    hsv = rgb_to_hsv_u01(rgb_u8)
    hch = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    x = rgb_u8.astype(np.float32) / 255.0
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    redness = (r - 0.5 * (g + b)).astype(np.float32)  # [-1,1] roughly
    ycbcr = rgb_to_ycbcr_u8(rgb_u8).astype(np.float32)
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    w_eye = np.zeros((h, w), dtype=np.float32)
    w_lip = np.zeros((h, w), dtype=np.float32)

    for _, fy0, fx0, fy1, fx1 in face_boxes:
        fh = max(1, fy1 - fy0)
        fw = max(1, fx1 - fx0)
        midx = (fx0 + fx1) // 2

        # Face interior soft region = dilated skin inside bbox (continuous)
        skin_roi = skin_bin[fy0:fy1, fx0:fx1].astype(np.float32)
        dil_r = max(3, int(0.06 * min(fh, fw)))
        face_soft = _box_blur_gray_f32(skin_roi, radius=dil_r)
        face_soft = np.clip(face_soft / (face_soft.max() + 1e-6), 0.0, 1.0)

        # Eye band (upper-mid)
        ey0 = int(np.clip(fy0 + 0.22 * fh, 0, h))
        ey1 = int(np.clip(fy0 + 0.48 * fh, 0, h))
        ex0 = int(np.clip(fx0 + 0.10 * fw, 0, w))
        ex1 = int(np.clip(fx1 - 0.10 * fw, 0, w))
        if ey1 > ey0 and ex1 > ex0:
            fs = face_soft[(ey0 - fy0) : (ey1 - fy0), (ex0 - fx0) : (ex1 - fx0)]
            # --- Borrowed design: EyeMapC + EyeMapL in YCbCr (Hsu et al., 2002 style) ---
            y_roi = Y[ey0:ey1, ex0:ex1]
            cb_roi = Cb[ey0:ey1, ex0:ex1]
            cr_roi = Cr[ey0:ey1, ex0:ex1]

            # EyeMapC highlights eye chroma: high Cb, low Cr
            eps = 1e-6
            emc = (cb_roi * cb_roi + (255.0 - cr_roi) * (255.0 - cr_roi) + (cb_roi / (cr_roi + eps))) / 3.0
            emc = emc / (emc.max() + eps)

            # EyeMapL uses local contrast on luminance: dilate(Y) / (erode(Y)+1)
            # Use small max/min filter (fast in numpy) for robustness.
            r_l = 3
            dil = _max_filter_gray_f32(y_roi, radius=r_l)
            ero = _min_filter_gray_f32(y_roi, radius=r_l)
            eml = (dil + 1.0) / (ero + 1.0)
            eml = eml / (eml.max() + eps)

            eye_map = (emc * eml).astype(np.float32)
            eye_map *= np.clip(fs, 0.0, 1.0)

            # Multi-scale smoothing (kernel sizes) to produce continuous weights
            rs = int(min(max(6, eye_kernel_r_small), 0.22 * min(fh, fw)))
            rm = int(min(max(rs + 2, eye_kernel_r_mid), 0.28 * min(fh, fw)))
            rl = int(min(max(rm + 2, eye_kernel_r_large), 0.34 * min(fh, fw)))
            resp = np.maximum.reduce(
                [
                    _box_blur_gray_f32(eye_map, radius=max(2, rs)),
                    _box_blur_gray_f32(eye_map, radius=max(2, rm)),
                    _box_blur_gray_f32(eye_map, radius=max(2, rl)),
                ]
            )
            # relaxed normalization to be more permissive across images
            lo = float(np.percentile(resp, 60.0))
            hi = float(np.percentile(resp, 98.5))
            ww = np.clip((resp - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
            w_eye[ey0:ey1, ex0:ex1] = np.maximum(w_eye[ey0:ey1, ex0:ex1], ww)

        # Lip band (lower)
        ly0 = int(np.clip(fy0 + 0.62 * fh, 0, h))
        ly1 = int(np.clip(fy0 + 0.88 * fh, 0, h))
        lx0 = int(np.clip(fx0 + 0.16 * fw, 0, w))
        lx1 = int(np.clip(fx1 - 0.16 * fw, 0, w))
        if ly1 > ly0 and lx1 > lx0:
            fs = face_soft[(ly0 - fy0) : (ly1 - fy0), (lx0 - fx0) : (lx1 - fx0)]
            rr = redness[ly0:ly1, lx0:lx1]
            ll = luma[ly0:ly1, lx0:lx1]
            cb_roi = Cb[ly0:ly1, lx0:lx1]
            cr_roi = Cr[ly0:ly1, lx0:lx1]

            # center prior: lips are near center horizontally
            xx = np.arange(lx0, lx1, dtype=np.float32)[None, :]
            cx = float(midx)
            sigma = max(6.0, 0.22 * fw)
            center = np.exp(-((xx - cx) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

            # --- Borrowed design: MouthMap in YCbCr (Cr high, Cb low) ---
            eps = 1e-6
            cr2 = cr_roi * cr_roi
            cr_over_cb = cr_roi / (cb_roi + eps)
            eta = 0.95 * float(np.mean(cr2[fs > 0.15]) / (np.mean(cr_over_cb[fs > 0.15]) + eps)) if np.any(fs > 0.15) else 0.95
            mouth_map = cr2 * (cr2 - eta * cr_over_cb) * (cr2 - eta * cr_over_cb)
            mouth_map = mouth_map / (mouth_map.max() + eps)

            mouth_map = mouth_map.astype(np.float32) * np.clip(fs, 0.0, 1.0) * center
            # suppress teeth/specular by high luma
            mouth_map *= (1.0 - _sigmoid((ll - 0.85) * 12.0))

            rs = int(min(max(6, lip_kernel_r_small), 0.20 * min(fh, fw)))
            rm = int(min(max(rs + 2, lip_kernel_r_mid), 0.26 * min(fh, fw)))
            rl = int(min(max(rm + 2, lip_kernel_r_large), 0.32 * min(fh, fw)))
            resp = np.maximum.reduce(
                [
                    _box_blur_gray_f32(mouth_map, radius=max(2, rs)),
                    _box_blur_gray_f32(mouth_map, radius=max(2, rm)),
                    _box_blur_gray_f32(mouth_map, radius=max(2, rl)),
                ]
            )
            lo = float(np.percentile(resp, 55.0))
            hi = float(np.percentile(resp, 98.5))
            ww = np.clip((resp - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
            w_lip[ly0:ly1, lx0:lx1] = np.maximum(w_lip[ly0:ly1, lx0:lx1], ww)

    # Light smoothing to avoid speckles
    w_eye = np.clip(_box_blur_gray_f32(w_eye, radius=4), 0.0, 1.0)
    w_lip = np.clip(_box_blur_gray_f32(w_lip, radius=4), 0.0, 1.0)
    return w_eye, w_lip


def _weighted_smooth_rgb_u8(rgb_u8: np.ndarray, weight_u01: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(weight_u01 > 1e-4):
        return rgb_u8
    blur = _box_blur_rgb_u8(rgb_u8, radius=int(radius))
    return _blend(rgb_u8, blur, np.clip(weight_u01, 0.0, 1.0))

def enhance_eyes_and_lips_soft(
    rgb_u8: np.ndarray,
    eye_weight_u01: np.ndarray,
    lip_weight_u01: np.ndarray,
    params: Optional[FeatureEnhanceParams] = None,
    *,
    eye_strength: float = 1.0,
    lip_strength: float = 1.0,
) -> np.ndarray:
    """
    Smooth color enhancement for eyes/lips: single HSV pass with spatially smooth
    weights (after ``smooth_feature_weights_for_enhancement``). No gamma/unsharp
    branch blends — avoids halos and keeps color change spatially continuous.
    """
    p = params or FeatureEnhanceParams()
    es = float(max(0.0, eye_strength))
    ls = float(max(0.0, lip_strength))
    if es < 1e-6 and ls < 1e-6:
        return rgb_u8

    we = np.clip(eye_weight_u01.astype(np.float32) * es, 0.0, 1.0)
    wl = np.clip(lip_weight_u01.astype(np.float32) * ls, 0.0, 1.0)
    if not (np.any(we > 1e-4) or np.any(wl > 1e-4)):
        return rgb_u8

    has_eye = bool(np.any(we > 1e-4))
    has_lip = bool(np.any(wl > 1e-4))

    hsv = rgb_to_hsv_u01(rgb_u8)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    hi_prot = np.clip((v - 0.80) / 0.20, 0.0, 1.0)

    eye_sg = float(np.clip(p.eye_s_gain, 0.2, 1.0))
    dv_eye_raw = we * 0.115 * (1.0 - v) * (1.0 - 0.30 * s) * (1.0 - 0.86 * hi_prot)
    dv_eye = np.tanh(dv_eye_raw * 5.6) * 0.14

    # Smooth, continuous eye-local contrast curve on V (no sharpening; preserves detail)
    ecs = float(np.clip(getattr(p, "eye_contrast_strength", 0.75), 0.0, 1.2))
    ecg = float(np.clip(getattr(p, "eye_contrast_gain", 2.8), 0.6, 6.0))
    if ecs > 1e-6 and has_eye:
        denom = float(np.tanh(0.5 * ecg) + 1e-6)
        v_curve = 0.5 + 0.5 * (np.tanh((v - 0.5) * ecg) / denom)
        dv_con = (v_curve - v) * we * ecs * (1.0 - 0.90 * hi_prot)
        # limit to avoid washed highlights / clipped shadows
        dv_con = np.tanh(dv_con * 6.5) * 0.10
    else:
        dv_con = 0.0

    # Smooth, continuous *local* contrast inside eye region (high-pass on V)
    elcs = float(np.clip(getattr(p, "eye_local_contrast_strength", 0.85), 0.0, 2.0))
    elr = int(np.clip(getattr(p, "eye_local_contrast_radius", 3), 1, 9))
    ellim = float(np.clip(getattr(p, "eye_local_contrast_limit", 0.10), 0.02, 0.22))
    if elcs > 1e-6 and has_eye:
        v_blur = _box_blur_gray_f32(v, radius=elr)
        detail = v - v_blur
        # limit detail push to avoid halos / noise boost
        dv_local = np.tanh((detail * elcs) / (ellim + 1e-6)) * ellim
        dv_local = dv_local * we * (1.0 - 0.90 * hi_prot)
    else:
        dv_local = 0.0

    s_mul_eye = 1.0 - we * (1.0 - eye_sg)
    # Allow slight saturation increase but avoid overly saturated highlights.
    s_mul_eye = np.clip(s_mul_eye, 0.72, 1.15)

    dh_raw = wl * (float(p.lip_hue_shift) / 360.0)
    dh = np.tanh(dh_raw * 24.0) / 24.0

    lip_sg = float(np.clip(p.lip_s_gain, 0.6, 2.2))
    s_boost = wl * (lip_sg - 1.0) * 0.68
    s_boost = np.tanh(s_boost * 2.2) * 0.26
    s_mul_lip = 1.0 + s_boost

    dv_lip_raw = wl * 0.055 * (1.0 - v) * (1.0 - 0.82 * hi_prot)
    dv_lip = np.tanh(dv_lip_raw * 5.6) * 0.11

    h2 = (h + dh) % 1.0
    s2 = np.clip(s * s_mul_eye * s_mul_lip, 0.0, 1.0)
    v2 = np.clip(v + dv_eye + dv_con + dv_local + dv_lip, 0.0, 0.999)

    return hsv_u01_to_rgb_u8(np.stack([h2, s2, v2], axis=-1))


def enhance_beauty(
    rgb_u8: np.ndarray,
    *,
    skin_mask_u8: Optional[np.ndarray] = None,
    skin_params: Optional[SkinToneParams] = None,
    feature_params: Optional[FeatureEnhanceParams] = None,
    pseudocolor_cmap: str = "jet",
    skin_strength: float = 1.0,
    eye_strength: float = 1.0,
    lip_strength: float = 1.0,
    use_explicit_eye_lip_masks: bool = False,
) -> EnhanceOutputs:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be (H,W,3) uint8")

    # 1) Skin mask (prefer Member1 segmentation API)
    if skin_mask_u8 is None:
        skin_mask_u8 = None
        try:
            _seg = _import_segmentation_with_stub()
            skin_mask_u8, _ = _seg.segment_v4(rgb_u8)
        except Exception:
            skin_mask_u8 = ycbcr_skin_mask_u8_adaptive(rgb_u8)
    else:
        if skin_mask_u8.ndim == 3:
            skin_mask_u8 = skin_mask_u8[:, :, 0]
        skin_mask_u8 = skin_mask_u8.astype(np.uint8, copy=False)

    # 2) Whitening + smoothing
    sp = skin_params or SkinToneParams()
    ss = float(max(0.0, skin_strength))
    if ss <= 1e-6:
        cur = rgb_u8
    else:
        cur = smooth_skin_color_enhance(rgb_u8, skin_mask_u8, sp, skin_strength=ss)
        # Optional legacy磨皮 (off by default: smooth_strength=0)
        sm = float(np.clip(sp.smooth_strength * min(1.5, ss), 0.0, 1.0))
        if sm > 1e-6:
            cur = whiten_and_smooth_skin(
                cur,
                skin_mask_u8,
                whiten_strength=float(np.clip(sp.whiten_strength * min(1.5, ss), 0.0, 1.25)) * 0.35,
                smooth_strength=sm,
                smooth_radius=max(1, int(sp.smooth_radius)),
                detail_amount=float(np.clip(sp.detail_amount, 0.0, 1.0)),
                feather_radius=int(sp.feather_radius),
            )

    # 4) Estimate eyes/lips and enhance local contrast
    regions = estimate_face_regions_from_skin_mask(skin_mask_u8)
    if use_explicit_eye_lip_masks:
        eye_mask_u8, lip_mask_u8 = estimate_eye_lip_masks_u8(
            cur, skin_mask_u8, regions=regions, feather_radius=int((feature_params or FeatureEnhanceParams()).feather_radius)
        )
        cur = enhance_eyes_and_lips(
            cur,
            eye_mask_u8,
            lip_mask_u8,
            params=feature_params,
            eye_strength=float(eye_strength),
            lip_strength=float(lip_strength),
        )
    else:
        fp = feature_params or FeatureEnhanceParams()
        w_eye, w_lip = _soft_eye_lip_weights(
            cur,
            skin_mask_u8,
            eye_kernel_r_small=int(fp.eye_kernel_r_small),
            eye_kernel_r_mid=int(fp.eye_kernel_r_mid),
            eye_kernel_r_large=int(fp.eye_kernel_r_large),
            lip_kernel_r_small=int(fp.lip_kernel_r_small),
            lip_kernel_r_mid=int(fp.lip_kernel_r_mid),
            lip_kernel_r_large=int(fp.lip_kernel_r_large),
        )
        hh, ww = cur.shape[0], cur.shape[1]
        w_eye = smooth_feature_weights_for_enhancement(w_eye, fp, image_hw=(hh, ww))
        w_lip = smooth_feature_weights_for_enhancement(w_lip, fp, image_hw=(hh, ww))
        w_eye = contrast_enhance_feature_weights(w_eye, fp)
        w_lip = contrast_enhance_feature_weights(w_lip, fp)
        cur = enhance_eyes_and_lips_soft(
            cur,
            w_eye,
            w_lip,
            params=fp,
            eye_strength=float(eye_strength),
            lip_strength=float(lip_strength),
        )
        # keep outputs for compatibility (all zeros in maskless mode)
        eye_mask_u8 = np.zeros(rgb_u8.shape[:2], dtype=np.uint8)
        lip_mask_u8 = np.zeros(rgb_u8.shape[:2], dtype=np.uint8)

    # 4.5) Optional post-smooth (default 0 — avoids blur / lost detail)
    if ss > 1e-6:
        pst = float(np.clip((skin_params or SkinToneParams()).post_smooth_strength, 0.0, 1.0))
        if pst > 1e-6 and np.any(skin_mask_u8 > 127):
            alpha = _feather_mask_u01(skin_mask_u8, radius=max(8, int((skin_params or SkinToneParams()).feather_radius // 2)))
            alpha = _suppress_highlights_alpha(cur, alpha, thr=0.92)
            alpha = np.clip(alpha * pst, 0.0, 1.0)
            blur = _box_blur_rgb_u8(cur, radius=1)
            cur = _blend(cur, blur, alpha)

    # 5) Pseudo-color (on pseudo-luma)
    x = cur.astype(np.float32)
    pseudo_luma = (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2])
    pseudo_luma_u8 = np.clip(pseudo_luma, 0, 255).astype(np.uint8)
    pseudocolor_rgb_u8 = apply_pseudocolor(pseudo_luma_u8, cmap=pseudocolor_cmap)

    return EnhanceOutputs(
        enhanced_rgb_u8=cur,
        skin_mask_u8=skin_mask_u8,
        eye_mask_u8=eye_mask_u8,
        lip_mask_u8=lip_mask_u8,
        pseudo_luma_u8=pseudo_luma_u8,
        pseudocolor_rgb_u8=pseudocolor_rgb_u8,
    )


# Backward-compat alias expected by some integrations
def enhance_beauty_simple(rgb_u8: np.ndarray) -> np.ndarray:
    return enhance_beauty(rgb_u8).enhanced_rgb_u8


# =========================
# I/O utilities (imported by Member5)
# =========================

def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _save_rgb(rgb_u8: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(rgb_u8).save(path)


def _save_mask(mask_u8: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if mask_u8.ndim == 3:
        m = mask_u8[:, :, 0]
    else:
        m = mask_u8
    Image.fromarray(m.astype(np.uint8)).save(path)


# Note: CLI demo removed. Use `code/demo_color_feature_enhance.py` instead.
