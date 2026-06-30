"""
Member 4: region-guided color processing for portrait images.

Processing chain (course-style IP pipeline):
  1) Color space conversion     — RGB <-> YCbCr, RGB <-> HSV
  2) Segmentation               — skin band + Otsu(Y) [Member 1], ROI heuristics
  3) Order-statistic features   — median Y/Cb/Cr on skin; ROI luminance percentiles
  4) Morphological ops          — binary dilation on label maps (box-filter SE)
  5) Spatial filtering          — separable mean (box) filter for masks / V channel
  6) Point operations           — HSV channel gains, highlight compression
  7) Image fusion               — smoothstep weights + alpha blending

Public entry: enhance_beauty_Z(rgb, roi_boxes, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from segmentation import rgb_to_ycbcr, segment_v4

RegionKind = Literal["eye", "lip"]

# =============================================================================
# Data structures (parameters & geometry)
# =============================================================================


@dataclass
class RegionBox:
    """Axis-aligned ROI in image coordinates (manual annotation)."""

    x0: int
    y0: int
    x1: int
    y1: int
    kind: RegionKind

    def clamp(self, w: int, h: int) -> "RegionBox":
        x0 = int(np.clip(min(self.x0, self.x1), 0, w - 1))
        x1 = int(np.clip(max(self.x0, self.x1), 0, w - 1))
        y0 = int(np.clip(min(self.y0, self.y1), 0, h - 1))
        y1 = int(np.clip(max(self.y0, self.y1), 0, h - 1))
        return RegionBox(x0, y0, x1, y1, self.kind)


@dataclass
class SkinStatsZ:
    """Robust skin-tone anchors (median order statistics on cautious skin mask)."""

    y_med: float
    cb_med: float
    cr_med: float


@dataclass
class SkinMaskParamsZ:
    """Segmentation / morphology parameters for skin label maps."""

    cautious_y_margin: float = 34.0
    cautious_y_abs_min: float = 46.0
    dilate_r: int = 2


@dataclass
class ColorDetectParamsZ:
    """Chroma / saturation / percentile thresholds inside eye and lip ROIs."""

    skin_cb_span: float = 13.0
    skin_cr_span: float = 15.0
    eye_gray_chroma: float = 22.0
    eye_max_sat: float = 0.30
    eye_white_chroma: float = 11.0
    eye_white_max_sat: float = 0.24
    eye_black_pctl: float = 13.0
    eye_white_pctl: float = 72.0
    eye_pupil_chroma_min: float = 5.0
    eye_skin_cb: float = 6.5
    eye_skin_cr: float = 7.5
    eye_skin_y: float = 9.0
    eye_skin_chroma_min: float = 5.0
    eye_flesh_cb: float = 16
    eye_flesh_chroma_max: float = 20
    eye_flesh_sat_min: float = 0.16
    eye_connect_r: int = 2
    lip_min_sat: float = 0.10
    lip_min_chroma: float = 4.0
    lip_skin_cb_span: float = 10.0
    lip_skin_cr_span: float = 9.0
    lip_skin_y_span: float = 10.0
    lip_skin_chroma_max: float = 22.0
    lip_tooth_y_above: float = 8.0
    lip_tooth_chroma: float = 18.0
    lip_tooth_max_sat: float = 0.28
    lip_tooth_red_max: float = 14.0
    lip_connect_r: int = 2


@dataclass
class DetectThZ:
    """Skin-tone–adaptive chroma / luminance gates (derived from SkinStatsZ)."""

    abs_black: float
    abs_white: float
    y_dark: float
    y_bright: float
    bright_margin: float
    sclera_keep_y: float
    flesh_cr_lo: float
    flesh_y: float
    lip_cr_above: float
    lip_red_above: float
    lip_cr_strong: float
    lip_red_strong: float
    lip_red_keep: float
    lip_cr_keep: float
    lip_skin_red_max: float
    lip_pink_cb: float
    lip_tooth_cr_cap: float


@dataclass
class EnhanceParamsZ:
    """HSV/RGB point gains and fusion strengths per tissue label."""

    feather: int = 11
    skin_strength: float = 1.0
    skin_whiten: float = 1.12
    skin_v_gain: float = 1.28
    skin_s_gain: float = 0.88
    skin_hue_shift_deg: float = -1.5
    skin_highlight_thr: float = 0.90
    skin_rgb_warmth: float = 0.38
    skin_v_whiten_cap: float = 0.24
    skin_v_w_scale: float = 0.22
    skin_midtone_pull: float = 0.50
    skin_rgb_lift: float = 16.0
    eye_strength: float = 1.20
    eye_v_lift: float = 0.16
    eye_contrast: float = 0.44
    eye_contrast_tanh: float = 2.6
    eye_sclera_v0: float = 0.30
    eye_sclera_span: float = 0.18
    eye_sat: float = 0.86
    lip_strength: float = 1.0
    lip_sat: float = 1.50
    lip_v_lift: float = 0.07
    lip_hue_shift: float = 0.016


@dataclass
class EnhanceOutputsZ:
    enhanced_rgb_u8: np.ndarray
    skin_mask_u8: np.ndarray
    eye_mask_u8: np.ndarray
    lip_mask_u8: np.ndarray


# =============================================================================
# Stage A — spatial-domain linear filtering (separable box / mean filter)
# =============================================================================


def spatial_mean_filter(channel: np.ndarray, radius: int) -> np.ndarray:
    """O(1) separable mean filter via summed-area table (same as legacy _box_blur)."""
    r = int(radius)
    if r <= 0:
        return channel.astype(np.float32)
    h, w = channel.shape
    p = np.pad(channel.astype(np.float32), ((r, r), (r, r)), mode="edge")
    integral = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.float32)
    integral[1:, 1:] = p.cumsum(0).cumsum(1)
    y0, y1 = np.arange(h), np.arange(h) + 2 * r + 1
    x0, x1 = np.arange(w), np.arange(w) + 2 * r + 1
    a = integral[y0[:, None], x0[None, :]]
    b = integral[y0[:, None], x1[None, :]]
    c = integral[y1[:, None], x0[None, :]]
    d = integral[y1[:, None], x1[None, :]]
    return (d - b - c + a) / float((2 * r + 1) ** 2)


def feather_alpha_map(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    """Soft weight in [0,1]: mean-filter binary mask (alpha matting support)."""
    return np.clip(spatial_mean_filter(mask_u8.astype(np.float32) / 255.0, max(1, int(radius))), 0, 1)


def smoothstep_weight_map(weights: np.ndarray) -> np.ndarray:
    """Per-pixel smoothstep on [0,1] for C1-soft fusion boundaries."""
    w = np.clip(weights.astype(np.float32), 0, 1)
    return w * w * (3.0 - 2.0 * w)


# =============================================================================
# Stage B — morphology (binary dilation approximated with mean filter + threshold)
# =============================================================================


def morphological_dilation_u8(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    """Flat SE dilation: box-filter foreground then threshold (legacy _dilate_u8)."""
    r = int(max(0, radius))
    if r == 0:
        return (mask_u8 > 127).astype(np.uint8) * 255
    fg = spatial_mean_filter((mask_u8 > 127).astype(np.float32), r)
    return (fg > 0.2).astype(np.uint8) * 255


# =============================================================================
# Stage C — color spaces & chrominance features
# =============================================================================


def chroma_magnitude_cb_cr(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Chroma magnitude in YCbCr around neutral (Cb,Cr)=(128,128)."""
    return np.sqrt((cb - 128.0) ** 2 + (cr - 128.0) ** 2)


def rgb_saturation_map(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel saturation (max-min)/max in normalized RGB."""
    x = rgb.astype(np.float32) / 255.0
    cmax, cmin = x.max(axis=-1), x.min(axis=-1)
    return np.where(cmax > 1e-5, (cmax - cmin) / (cmax + 1e-6), 0.0)


def rgb_to_hsv_planes(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose RGB uint8 to H,S,V in [0,1]."""
    x = rgb_u8.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    d = cmax - cmin + 1e-6
    h = np.zeros_like(cmax)
    m = cmax == r
    h[m] = ((g - b)[m] / d[m]) % 6
    m = cmax == g
    h[m] = (b - r)[m] / d[m] + 2
    m = cmax == b
    h[m] = (r - g)[m] / d[m] + 4
    return (h / 6.0) % 1.0, np.where(cmax > 1e-5, d / (cmax + 1e-6), 0.0), cmax


def hsv_planes_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compose H,S,V (float [0,1]) back to RGB uint8."""
    i = np.floor(h * 6).astype(np.int32) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros((*h.shape, 3), np.float32)
    for k, (a, b, c) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m, 0], out[m, 1], out[m, 2] = a[m], b[m], c[m]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def extract_roi_ycbcr_planes(
    ycbcr: np.ndarray, box: RegionBox
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop Y, Cb, Cr planes inside clamped ROI."""
    return (
        ycbcr[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1, 0].astype(np.float32),
        ycbcr[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1, 1].astype(np.float32),
        ycbcr[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1, 2].astype(np.float32),
    )


# =============================================================================
# Stage D — segmentation: skin label maps (histogram / Otsu via Member 1)
# =============================================================================


def segment_skin_histogram_otsu(rgb_u8: np.ndarray) -> np.ndarray:
    """Member 1: YCbCr skin chroma band + global Otsu threshold on Y."""
    mask, _ = segment_v4(rgb_u8)
    return mask


def compute_skin_cautious_Z(rgb_u8: np.ndarray, p: Optional[SkinMaskParamsZ] = None) -> np.ndarray:
    """
    Detection skin mask: segment_v4 then luminance gate
    Y >= max(Y_min, median(Y|skin) - margin)  [robust to shadows].
    """
    p = p or SkinMaskParamsZ()
    base = segment_skin_histogram_otsu(rgb_u8)
    m = base > 127
    if np.any(m):
        y = rgb_to_ycbcr(rgb_u8).astype(np.float32)[:, :, 0]
        y_med = float(np.median(y[m]))
        m &= y >= max(p.cautious_y_abs_min, y_med - p.cautious_y_margin)
    return (m.astype(np.uint8) * 255)


def compute_skin_mask_Z(rgb_u8: np.ndarray, p: Optional[SkinMaskParamsZ] = None) -> np.ndarray:
    """Enhancement skin: segment_v4 (+ optional morphological dilation)."""
    p = p or SkinMaskParamsZ()
    base = segment_skin_histogram_otsu(rgb_u8)
    m = base > 127
    if int(p.dilate_r) > 0:
        m = morphological_dilation_u8((m.astype(np.uint8) * 255), int(p.dilate_r)) > 127
    return (m.astype(np.uint8) * 255)


# =============================================================================
# Stage E — order statistics & adaptive thresholds
# =============================================================================


def skin_stats_from_mask(rgb_u8: np.ndarray, skin_mask_u8: np.ndarray) -> SkinStatsZ:
    """Median Y, Cb, Cr on cautious skin (50th percentile / order statistics)."""
    ycc = rgb_to_ycbcr(rgb_u8)
    m = skin_mask_u8 > 127
    if not np.any(m):
        y = ycc[:, :, 0].astype(np.float32)
        return SkinStatsZ(float(np.median(y)), 112.0, 152.0)
    return SkinStatsZ(
        float(np.median(ycc[:, :, 0][m])),
        float(np.median(ycc[:, :, 1][m])),
        float(np.median(ycc[:, :, 2][m])),
    )


def derive_adaptive_chroma_thresholds(st: SkinStatsZ, p: ColorDetectParamsZ) -> DetectThZ:
    """Map skin tone (warm/light scalars) to eye/lip chroma and luminance gates."""
    y, cr = st.y_med, st.cr_med
    warm = float(np.clip((cr - 142.0) / 22.0, -0.4, 1.2))
    light = float(np.clip((y - 115.0) / 38.0, -0.4, 1.2))
    cr_gap = float(np.clip((158.0 - cr) * 0.35, 0.0, 10.0))
    lip_cr = 5.0 + cr_gap + warm * 3.0
    lip_red = float(np.clip(0.034 + warm * 0.013 + cr_gap * 0.001, 0.030, 0.068))
    return DetectThZ(
        abs_black=float(np.clip(y * 0.36 + 6.0, 42.0, 62.0)),
        abs_white=float(np.clip(y + 14.0 + light * 8.0, 132.0, 175.0)),
        y_dark=18.0 + light * 8.0,
        y_bright=6.0 + warm * 2.0,
        bright_margin=7.0 + light * 3.0,
        sclera_keep_y=9.0 + light * 4.0,
        flesh_cr_lo=6.0 + warm * 3.0,
        flesh_y=14.0 + light * 4.0,
        lip_cr_above=lip_cr,
        lip_red_above=lip_red,
        lip_cr_strong=9.0 + warm * 4.0,
        lip_red_strong=float(np.clip(0.032 + warm * 0.010, 0.028, 0.056)),
        lip_red_keep=lip_red * 1.05,
        lip_cr_keep=lip_cr + 4.0,
        lip_skin_red_max=float(np.clip(0.038 + warm * 0.009, 0.034, 0.056)),
        lip_pink_cb=2.5 + warm * 2.0,
        lip_tooth_cr_cap=4.0 + warm * 2.0,
    )


def luminance_percentile_threshold(
    y_roi: np.ndarray, percentile: float, fallback: float, *, min_samples: int = 12
) -> float:
    """ROI luminance threshold from empirical CDF (histogram percentile)."""
    if y_roi.size >= min_samples:
        return float(np.percentile(y_roi, percentile))
    return fallback


def fuse_dark_bright_thresholds(
    global_th: float, cdf_th: float, *, mode: Literal["dark", "bright"]
) -> float:
    """Combine global skin-adaptive gate with local ROI percentile gate."""
    if mode == "dark":
        return min(global_th, cdf_th)
    return max(global_th, cdf_th)


# =============================================================================
# Stage F — ROI binary classification (eye / lip label maps)
# =============================================================================


def _eye_gray_ok(ch: np.ndarray, sat: np.ndarray, p: ColorDetectParamsZ) -> np.ndarray:
    return (ch <= float(p.eye_gray_chroma)) & (sat <= float(p.eye_max_sat))


def _eye_white_ok(ch: np.ndarray, sat: np.ndarray, p: ColorDetectParamsZ) -> np.ndarray:
    return (ch <= float(p.eye_white_chroma)) & (sat <= float(p.eye_white_max_sat))


def _eye_flesh_like(y, cb, cr, ch, sat, st: SkinStatsZ, p: ColorDetectParamsZ, th: DetectThZ) -> np.ndarray:
    return (
        (ch >= float(p.eye_skin_chroma_min))
        & (ch <= float(p.eye_flesh_chroma_max))
        & (sat >= float(p.eye_flesh_sat_min))
        & (np.abs(cb - st.cb_med) <= float(p.eye_flesh_cb))
        & (cr >= st.cr_med - th.flesh_cr_lo)
        & (np.abs(y - st.y_med) <= th.flesh_y)
    )


def _eye_skin_ex(y, cb, cr, ch, sat, skin_roi, st: SkinStatsZ, p: ColorDetectParamsZ, th: DetectThZ) -> np.ndarray:
    keep = _eye_gray_ok(ch, sat, p) & (y >= st.y_med + th.sclera_keep_y)
    tint = (
        (np.abs(cb - st.cb_med) <= p.eye_skin_cb)
        & (np.abs(cr - st.cr_med) <= p.eye_skin_cr)
        & (np.abs(y - st.y_med) <= p.eye_skin_y)
        & (ch >= float(p.eye_skin_chroma_min))
    )
    flesh = _eye_flesh_like(y, cb, cr, ch, sat, st, p, th) & (~keep)
    return tint | flesh | ((skin_roi > 127) & (~keep))


def _lip_skin_ex(y, cb, cr, ch, sat, red, skin_roi, st: SkinStatsZ, p: ColorDetectParamsZ, th: DetectThZ) -> np.ndarray:
    keep = (
        (cr >= st.cr_med + th.lip_cr_keep)
        & (red >= th.lip_red_keep * 255.0)
        & (sat >= float(p.lip_min_sat))
        & (ch >= float(p.lip_min_chroma))
    )
    near = (
        (np.abs(cb - st.cb_med) <= p.lip_skin_cb_span)
        & (np.abs(cr - st.cr_med) <= p.lip_skin_cr_span)
        & (np.abs(y - st.y_med) <= p.lip_skin_y_span)
    )
    tint = near & (ch <= float(p.lip_skin_chroma_max)) & (red < th.lip_skin_red_max * 255.0)
    return (tint | (skin_roi > 127)) & (~keep)


def classify_roi_binary_mask(
    rgb_u8: np.ndarray,
    box: RegionBox,
    kind: RegionKind,
    st: SkinStatsZ,
    skin_u8: np.ndarray,
    p: ColorDetectParamsZ,
    th: DetectThZ,
) -> np.ndarray:
    """Per-ROI segmentation: thresholding + chroma rules -> {0,255} mask."""
    ycc = rgb_to_ycbcr(rgb_u8)
    y, cb, cr = extract_roi_ycbcr_planes(ycc, box)
    ch = chroma_magnitude_cb_cr(cb, cr)
    if kind == "eye":
        sat = rgb_saturation_map(rgb_u8[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1])
        skin_roi = skin_u8[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1] if skin_u8 is not None else np.zeros(y.shape, np.uint8)
        ex = _eye_skin_ex(y, cb, cr, ch, sat, skin_roi, st, p, th)
        gray_ok = _eye_gray_ok(ch, sat, p)
        white_ok = _eye_white_ok(ch, sat, p)
        not_flesh = ~_eye_flesh_like(y, cb, cr, ch, sat, st, p, th)
        med = float(np.median(y))
        tb = fuse_dark_bright_thresholds(
            th.abs_black,
            luminance_percentile_threshold(y, p.eye_black_pctl, th.abs_black),
            mode="dark",
        )
        tw = fuse_dark_bright_thresholds(
            th.abs_white,
            luminance_percentile_threshold(y, p.eye_white_pctl, th.abs_white),
            mode="bright",
        )
        dark = (y <= tb) | ((y <= st.y_med - th.y_dark) & (ch >= float(p.eye_pupil_chroma_min)))
        bright_y = (y >= tw) | (y >= st.y_med + th.y_bright) | (y >= med + th.bright_margin)
        is_white_y = (y >= tw) | (y >= st.y_med + th.y_bright + 8.0)
        purity = np.where(is_white_y, white_ok, gray_ok)
        bright = not_flesh & bright_y & purity
        return ((dark | bright) & (~ex)).astype(np.uint8) * 255
    rgb = rgb_u8[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    skin_roi = skin_u8[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1] if skin_u8 is not None else np.zeros(y.shape, np.uint8)
    red = rgb[:, :, 0].astype(np.float32) - 0.5 * (rgb[:, :, 1] + rgb[:, :, 2])
    sat = rgb_saturation_map(rgb)
    lip_pure = (
        (sat >= float(p.lip_min_sat))
        & (ch >= float(p.lip_min_chroma))
        & (red >= th.lip_red_above * 255.0)
        & (cr >= cb + th.lip_pink_cb)
    )
    cr_ok = cr >= st.cr_med + th.lip_cr_above
    pink_ok = (
        (sat >= float(p.lip_min_sat))
        & (ch >= float(p.lip_min_chroma))
        & (cr >= st.cr_med + max(4.0, th.lip_cr_above - 2.0))
        & (cr >= cb + max(1.5, th.lip_pink_cb - 1.0))
        & (red >= th.lip_red_above * 0.86 * 255.0)
    )
    lip_hit = (cr_ok & lip_pure) | pink_ok | (
        (cr >= st.cr_med + th.lip_cr_strong)
        & (red >= th.lip_red_strong * 255.0)
        & (sat >= float(p.lip_min_sat))
        & (ch >= float(p.lip_min_chroma))
    )
    ex = _lip_skin_ex(y, cb, cr, ch, sat, red, skin_roi, st, p, th)
    tooth = (
        (y >= st.y_med + p.lip_tooth_y_above)
        & (ch <= p.lip_tooth_chroma)
        & (cr < st.cr_med + th.lip_tooth_cr_cap)
        & (sat <= p.lip_tooth_max_sat)
        & (red < float(p.lip_tooth_red_max))
    )
    return (lip_hit & (~ex) & (~tooth)).astype(np.uint8) * 255


def build_eye_lip_masks_Z(
    rgb_u8: np.ndarray,
    boxes: Sequence[RegionBox],
    skin_cautious_u8: np.ndarray,
    params: Optional[ColorDetectParamsZ] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse per-ROI masks; post-process with morphological dilation and leak removal."""
    p = params or ColorDetectParamsZ()
    st = skin_stats_from_mask(rgb_u8, skin_cautious_u8)
    th = derive_adaptive_chroma_thresholds(st, p)
    h, w = rgb_u8.shape[:2]
    eye = np.zeros((h, w), np.uint8)
    lip = np.zeros((h, w), np.uint8)
    for box in boxes:
        b = box.clamp(w, h)
        roi = classify_roi_binary_mask(rgb_u8, b, box.kind, st, skin_cautious_u8, p, th)
        if box.kind == "eye":
            eye[b.y0 : b.y1 + 1, b.x0 : b.x1 + 1] = np.maximum(eye[b.y0 : b.y1 + 1, b.x0 : b.x1 + 1], roi)
        else:
            lip[b.y0 : b.y1 + 1, b.x0 : b.x1 + 1] = np.maximum(lip[b.y0 : b.y1 + 1, b.x0 : b.x1 + 1], roi)
    if np.any(eye > 127):
        ycc = rgb_to_ycbcr(rgb_u8).astype(np.float32)
        leak = (eye > 127) & _eye_skin_ex(
            ycc[:, :, 0],
            ycc[:, :, 1],
            ycc[:, :, 2],
            chroma_magnitude_cb_cr(ycc[:, :, 1], ycc[:, :, 2]),
            rgb_saturation_map(rgb_u8),
            skin_cautious_u8,
            st,
            p,
            th,
        )
        eye[leak] = 0
        eye = morphological_dilation_u8(eye, int(p.eye_connect_r))
    if np.any(lip > 127):
        ycc = rgb_to_ycbcr(rgb_u8).astype(np.float32)
        ch = chroma_magnitude_cb_cr(ycc[:, :, 1], ycc[:, :, 2])
        red = rgb_u8[:, :, 0].astype(np.float32) - 0.5 * (
            rgb_u8[:, :, 1].astype(np.float32) + rgb_u8[:, :, 2].astype(np.float32)
        )
        for dilate in (False, True):
            if dilate:
                lip = morphological_dilation_u8(lip, int(p.lip_connect_r))
            lip[(lip > 127) & _lip_skin_ex(ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2], ch, rgb_saturation_map(rgb_u8), red, skin_cautious_u8, st, p, th)] = 0
    return eye, lip


# =============================================================================
# Stage G — HSV point operations & multi-label fusion
# =============================================================================


def apply_skin_tone_mapping(rgb_u8: np.ndarray, alpha: np.ndarray, p: EnhanceParamsZ) -> np.ndarray:
    """Skin branch: HSV gains + RGB bias, fused with smoothstep(alpha)."""
    a = smoothstep_weight_map(np.clip(alpha.astype(np.float32), 0, 1))
    if not np.any(a > 1e-4):
        return rgb_u8
    x = rgb_u8.astype(np.float32) / 255.0
    luma = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    a *= np.where(luma > p.skin_highlight_thr, np.clip(1.0 - (luma - p.skin_highlight_thr) / 0.1, 0.2, 1.0), 1.0)
    h, s, v = rgb_to_hsv_planes(rgb_u8)
    out = hsv_planes_to_rgb(
        (h + a * (p.skin_hue_shift_deg / 360.0)) % 1.0,
        np.clip(s + a * np.tanh(s * (p.skin_s_gain - 1.0) * 3.0) * 0.38, 0, 1),
        np.clip(
            v
            + a
            * (
                np.tanh(v * (p.skin_v_gain - 1.0) * 2.8) * 0.42
                + np.minimum(p.skin_whiten * p.skin_v_w_scale * (1 - v), p.skin_v_whiten_cap)
            )
            + a * p.skin_midtone_pull * (np.clip(v + 0.12, 0, 0.92) - v) * 0.5,
            0,
            0.998,
        ),
    ).astype(np.float32)
    warm = rgb_u8.astype(np.float32) * (1.03, 1.01, 0.985) + (2.0, 1.0, 0.0)
    aw = smoothstep_weight_map(a) * p.skin_rgb_warmth
    out = out * (1 - aw[..., None]) + warm * aw[..., None]
    base = rgb_u8.astype(np.float32)
    return np.clip(base * (1 - a[..., None]) + out * a[..., None] + a[..., None] * p.skin_rgb_lift, 0, 255).astype(np.uint8)


def fuse_region_enhancements(
    rgb_u8: np.ndarray,
    skin_u8: np.ndarray,
    eye_u8: np.ndarray,
    lip_u8: np.ndarray,
    params: Optional[EnhanceParamsZ] = None,
) -> np.ndarray:
    """Sequential alpha fusion: skin -> eye (local V filter) -> lip on HSV planes."""
    p = params or EnhanceParamsZ()
    fr = int(p.feather) + 3
    _, _, v0 = rgb_to_hsv_planes(rgb_u8)
    out = apply_skin_tone_mapping(rgb_u8, smoothstep_weight_map(feather_alpha_map(skin_u8, fr)) * p.skin_strength, p)
    h, s, v = rgb_to_hsv_planes(out)
    we = smoothstep_weight_map(feather_alpha_map(eye_u8, fr)) * p.eye_strength
    wl = smoothstep_weight_map(feather_alpha_map(lip_u8, fr)) * p.lip_strength
    vb = spatial_mean_filter(v, 5)
    whi = smoothstep_weight_map(np.clip((v0 - p.eye_sclera_v0) / max(0.06, p.eye_sclera_span), 0, 1))
    v = np.clip(v + we * (whi * p.eye_v_lift + np.tanh((v - vb) * p.eye_contrast * p.eye_contrast_tanh) * 0.11), 0, 0.999)
    s = np.clip(s * (1.0 + we * (p.eye_sat - 1.0) * (0.35 + 0.65 * whi)), 0, 1)
    h = (h + wl * p.lip_hue_shift) % 1.0
    v = np.clip(v + wl * p.lip_v_lift, 0, 0.999)
    s = np.clip(s * (1.0 + wl * (p.lip_sat - 1.0)), 0, 1)
    return hsv_planes_to_rgb(h, s, v)


# =============================================================================
# Public API (unchanged names for pipeline / UI)
# =============================================================================


def enhance_Z(rgb_u8, skin_u8, eye_u8, lip_u8, params=None) -> np.ndarray:
    return fuse_region_enhancements(rgb_u8, skin_u8, eye_u8, lip_u8, params)


def build_panel_Z(
    rgb_u8, enhanced_u8, overlay_u8, *, boxes_u8: Optional[np.ndarray] = None, column_height=None
) -> np.ndarray:
    cols = [rgb_u8]
    if boxes_u8 is not None:
        cols.append(boxes_u8)
    cols.extend([overlay_u8, enhanced_u8])
    if column_height is None or column_height >= rgb_u8.shape[0]:
        return np.concatenate(cols, axis=1)
    from PIL import Image

    h, nw = int(column_height), max(1, int(round(rgb_u8.shape[1] * int(column_height) / float(rgb_u8.shape[0]))))
    col = lambda im: np.asarray(Image.fromarray(im).resize((nw, h), Image.Resampling.LANCZOS), dtype=np.uint8)
    return np.concatenate([col(im) for im in cols], axis=1)


def enhance_beauty_Z(
    rgb_u8: np.ndarray,
    boxes: Sequence[RegionBox],
    *,
    skin_mask_u8: Optional[np.ndarray] = None,
    skin_mask_params: Optional[SkinMaskParamsZ] = None,
    color_params: Optional[ColorDetectParamsZ] = None,
    enhance_params: Optional[EnhanceParamsZ] = None,
) -> EnhanceOutputsZ:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be (H,W,3)")
    ps = skin_mask_params or SkinMaskParamsZ()
    cautious = compute_skin_cautious_Z(rgb_u8, ps)
    loose = skin_mask_u8 if skin_mask_u8 is not None else compute_skin_mask_Z(rgb_u8, ps)
    eye, lip = build_eye_lip_masks_Z(rgb_u8, boxes, cautious, color_params)
    return EnhanceOutputsZ(fuse_region_enhancements(rgb_u8, loose, eye, lip, enhance_params), loose, eye, lip)


def overlay_masks(
    rgb_u8: np.ndarray,
    skin: np.ndarray,
    eye: np.ndarray,
    lip: np.ndarray,
    *,
    feather_skin: int = 12,
    feather_feature: int = 6,
    alpha_skin: float = 0.35,
    alpha_eye: float = 0.55,
    alpha_lip: float = 0.55,
) -> np.ndarray:
    """Visualize label maps (weighted color overlay for diagnostics)."""
    x = rgb_u8.astype(np.float32)
    for w, color, al in (
        (feather_alpha_map(skin, feather_skin), (0, 1, 0), alpha_skin),
        (feather_alpha_map(eye, feather_feature), (0, 1, 1), alpha_eye),
        (feather_alpha_map(lip, feather_feature), (1, 0, 1), alpha_lip),
    ):
        tint = np.stack([255.0 * color[0] * w, 255.0 * color[1] * w, 255.0 * color[2] * w], axis=-1)
        x = x * (1 - (w * al)[..., None]) + tint * (w * al)[..., None]
    return np.clip(x, 0, 255).astype(np.uint8)


# Backward-compatible aliases (identical behaviour; legacy names)
_box_blur = spatial_mean_filter
_smooth01 = smoothstep_weight_map
_dilate_u8 = morphological_dilation_u8
_chroma = chroma_magnitude_cb_cr
_rgb_sat = rgb_saturation_map
_rgb_to_hsv = rgb_to_hsv_planes
_hsv_to_rgb = hsv_planes_to_rgb
_detect_th = derive_adaptive_chroma_thresholds
_detect_roi = classify_roi_binary_mask
_apply_skin_tone_Z = apply_skin_tone_mapping
feather_u01 = feather_alpha_map
