"""
Member 2: Pimple removal on image + face-region mask for Member 3.

Pipeline:
  1. Detect eyes as LARGE mask holes (>= EYE_HOLE_MIN px); never modify them.
  2. Detect pimples: mask holes + RGB outliers (face skin distribution) before median.
  3. Median-filter pimple pixels on the original image only.
  4. Refine mask: fill all dots/holes (including eyes) for a solid Member 3 mask.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

INPUT_DIR = "./output/output_member1"
ORIGINAL_DIR = "./input_images"
OUTPUT_DIR = "./output/output_member2"
# Member 1 (segmentation.py) writes {basename}_mask.png and {basename}_mask_skin.png
MEMBER1_MASK_SUFFIX = "_mask.png"
MEMBER1_MASK_SKIN_SUFFIX = "_mask_skin.png"
KERNEL_SIZE = 3

# Pimple = small black dot; eyes = much larger holes (compare by size)
PIMPLE_HOLE_MIN = 1
PIMPLE_HOLE_MAX = 90

# Eyes / glasses: large enclosed holes — never median-filter or fill
EYE_HOLE_MIN = 100
EYE_REGION_Y_FRAC = 0.58  # upper fraction of face where large holes = eyes

# Nose / nostrils: medium holes in central face — not pimples
NOSE_HOLE_MIN = 15
NOSE_Y0_FRAC, NOSE_Y1_FRAC = 0.38, 0.72
NOSE_X0_FRAC, NOSE_X1_FRAC = 0.32, 0.68

# Face area: full head + chin; shirt collar excluded below exclude_shirt_frac
FACE_REGION_FRACTION = 0.95
EXCLUDE_SHIRT_FRAC = 0.86

MEDIAN_FILTER_SIZE = 9
MEDIAN_DEEP_KERNEL = 11       # stronger blur on saturated red pixels
MEDIAN_PASSES = 3
MEDIAN_DEEP_PASSES = 3
RGB_PIMPLE_MASK_DILATE = 4    # cover full pink patch before median
RGB_DEEP_DILATE = 2           # extra dilation for deep-red blobs

# RGB pink/red detection: learn distribution on face skin, flag statistical outliers
STAT_LOCAL_WIN = 15            # neighborhood for local redness vs neighbors
STAT_GLOBAL_Z = 2.4            # modified z (R-G, R-B, Cr) above trimmed skin stats
STAT_LOCAL_Z = 1.7             # local R-G excess vs 15x15 mean
STAT_LOCAL_DRG_MIN = 0.5
STAT_PERCENTILE_RG = 90          # on trimmed skin core (wider red/pink range)
STAT_PERCENTILE_DRG = 82         # fallback: local dR-G above this %ile
STAT_TRIM_HIGH_FRAC = 0.05       # exclude reddest 5% when fitting skin baseline
STAT_CHROMA_CHANNELS_MIN = 2   # redder than skin on at least N of (R-G, R-B, Cr)
STAT_DEEP_RED_EXCESS_PCT = 84  # saturated R - min(G,B) on face skin
STAT_DEEP_RG_PCT = 86          # R-G for deep red spots
RGB_BLOB_MIN = 2
RGB_BLOB_MAX = 450
# Lower face band to skip lips (nostrils already in nose_regions)
MOUTH_Y0_FRAC = 0.68
# Side bands in upper face (ears are naturally red in RGB)
EAR_X0_FRAC, EAR_X1_FRAC = 0.14, 0.86
EAR_Y1_FRAC = 0.62


def _label_binary(binary):
    if HAS_SCIPY:
        return ndimage.label(binary)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for i in range(h):
        for j in range(w):
            if not binary[i, j] or labels[i, j]:
                continue
            current += 1
            stack = [(i, j)]
            labels[i, j] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def _interior_holes(binary_fg, min_size, max_size):
    """Holes = background regions fully surrounded by foreground (black dots in mask)."""
    inv = ~binary_fg
    labeled, num_features = _label_binary(inv)
    holes = np.zeros_like(binary_fg, dtype=bool)

    for i in range(1, num_features + 1):
        region = labeled == i
        if np.any(region[0, :]) or np.any(region[-1, :]) or np.any(region[:, 0]) or np.any(region[:, -1]):
            continue
        size = int(region.sum())
        if min_size <= size <= max_size:
            holes[region] = True
    return holes


def _face_area_mask(binary_fg, fraction=FACE_REGION_FRACTION):
    """
    Face/neck band (forehead to chin). Below EXCLUDE_SHIRT_FRAC only keeps
    eroded face core so shirt/tie speckles are not treated as pimples.
    """
    ys, _ = np.where(binary_fg)
    if len(ys) == 0:
        return np.zeros_like(binary_fg, dtype=bool)
    y0, y1 = ys.min(), ys.max()
    h = max(y1 - y0, 1)
    region = np.zeros_like(binary_fg, dtype=bool)
    region[y0 : int(y0 + fraction * h), :] = True

    shirt_start = int(y0 + EXCLUDE_SHIRT_FRAC * h)
    if HAS_SCIPY:
        core = ndimage.binary_erosion(binary_fg, iterations=6)
    else:
        core = binary_fg.copy()
        for _ in range(6):
            pad = 1
            p = np.pad(core.astype(np.uint8), pad, mode="constant")
            h2, w2 = core.shape
            e = np.zeros_like(core)
            for i in range(h2):
                for j in range(w2):
                    e[i, j] = np.all(p[i : i + 3, j : j + 3])
            core = e.astype(bool)
    region[shirt_start:, :] &= core[shirt_start:, :]
    return region


def extract_eye_regions(binary_fg: np.ndarray) -> np.ndarray:
    """
    Preserve eyes: large black holes in the mask (much bigger than pimples).
    Uses enclosed holes >= EYE_HOLE_MIN in the upper face band.
    """
    ys, _ = np.where(binary_fg)
    if len(ys) == 0:
        return np.zeros_like(binary_fg, dtype=bool)

    y0, y1 = ys.min(), ys.max()
    upper_y = y0 + EYE_REGION_Y_FRAC * max(y1 - y0, 1)

    eyes = _interior_holes(binary_fg, EYE_HOLE_MIN, 999999)

    inv = ~binary_fg
    labeled, num_features = _label_binary(inv)
    for i in range(1, num_features + 1):
        region = labeled == i
        size = int(region.sum())
        if size < EYE_HOLE_MIN:
            continue
        cy = np.where(region)[0].mean()
        if cy <= upper_y:
            interior = not (
                np.any(region[0, :]) or np.any(region[-1, :])
                or np.any(region[:, 0]) or np.any(region[:, -1])
            )
            if interior:
                eyes[region] = True

    if HAS_SCIPY and np.any(eyes):
        eyes = ndimage.binary_dilation(eyes, iterations=2)
    return eyes


def _nose_area_mask(binary_fg: np.ndarray) -> np.ndarray:
    """Central face band where nostril / nose mask holes appear."""
    ys, xs = np.where(binary_fg)
    if len(ys) == 0:
        return np.zeros_like(binary_fg, dtype=bool)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    fh, fw = max(y1 - y0, 1), max(x1 - x0, 1)
    region = np.zeros_like(binary_fg, dtype=bool)
    region[
        int(y0 + NOSE_Y0_FRAC * fh) : int(y0 + NOSE_Y1_FRAC * fh),
        int(x0 + NOSE_X0_FRAC * fw) : int(x0 + NOSE_X1_FRAC * fw),
    ] = True
    return region


def extract_nose_regions(binary_fg: np.ndarray) -> np.ndarray:
    """
    Nose/nostril holes: medium black holes in the central face (not pimples).
    Smaller than eyes, larger than typical pimple speckles.
    """
    nose_area = _nose_area_mask(binary_fg)
    nose = _interior_holes(binary_fg, NOSE_HOLE_MIN, EYE_HOLE_MIN - 1)
    nose &= nose_area

    inv = ~binary_fg
    labeled, num_features = _label_binary(inv)
    h, w = binary_fg.shape
    for i in range(1, num_features + 1):
        region = labeled == i
        size = int(region.sum())
        if size < NOSE_HOLE_MIN or size >= EYE_HOLE_MIN:
            continue
        ys, xs = np.where(region)
        cy, cx = int(ys.mean()), int(xs.mean())
        if 0 <= cy < h and 0 <= cx < w and nose_area[cy, cx]:
            nose[region] = True

    if HAS_SCIPY and np.any(nose):
        nose = ndimage.binary_dilation(nose, iterations=1)
    return nose


def _small_black_components(binary_fg, face_area, min_size, max_size):
    """Small black connected components whose centroid lies in the face area."""
    inv = ~binary_fg
    labeled, num_features = _label_binary(inv)
    dots = np.zeros_like(binary_fg, dtype=bool)
    h, w = binary_fg.shape

    for i in range(1, num_features + 1):
        region = labeled == i
        size = int(region.sum())
        if not (min_size <= size <= max_size):
            continue
        ys, xs = np.where(region)
        cy, cx = int(ys.mean()), int(xs.mean())
        if not (0 <= cy < h and 0 <= cx < w and face_area[cy, cx]):
            continue
        touches_border = (
            np.any(region[0, :]) or np.any(region[-1, :])
            or np.any(region[:, 0]) or np.any(region[:, -1])
        )
        if touches_border and size > 500:
            continue
        dots[region] = True
    return dots


def _ear_exclusion_mask(binary_fg: np.ndarray) -> np.ndarray:
    """Upper side face bands where ear skin skews red/pink stats."""
    ys, xs = np.where(binary_fg)
    if len(ys) == 0:
        return np.zeros_like(binary_fg, dtype=bool)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    fh, fw = max(y1 - y0, 1), max(x1 - x0, 1)
    region = np.zeros_like(binary_fg, dtype=bool)
    y_hi = int(y0 + EAR_Y1_FRAC * fh)
    region[y0:y_hi, : int(x0 + EAR_X0_FRAC * fw)] = True
    region[y0:y_hi, int(x0 + EAR_X1_FRAC * fw) :] = True
    return region


def _mouth_exclusion_mask(binary_fg: np.ndarray) -> np.ndarray:
    """Lower-central face where lip redness should not count as pimples."""
    ys, xs = np.where(binary_fg)
    if len(ys) == 0:
        return np.zeros_like(binary_fg, dtype=bool)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    fh, fw = max(y1 - y0, 1), max(x1 - x0, 1)
    region = np.zeros_like(binary_fg, dtype=bool)
    region[int(y0 + MOUTH_Y0_FRAC * fh) :, int(x0 + 0.28 * fw) : int(x0 + 0.72 * fw)] = True
    return region


def _red_excess(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return r.astype(np.float32) - np.minimum(g.astype(np.float32), b.astype(np.float32))


def _median_blur_channel(channel: np.ndarray, k: int) -> np.ndarray:
    """Full-image median blur (odd k). Uses scipy, OpenCV, or a slow numpy fallback."""
    k = max(int(k) | 1, 3)
    ch = channel.astype(np.uint8)
    if HAS_SCIPY:
        return ndimage.median_filter(ch, size=k)
    if HAS_CV2:
        return cv2.medianBlur(ch, k)
    pad = k // 2
    padded = np.pad(ch.astype(np.float32), pad, mode="edge")
    h, w = ch.shape
    out = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            out[y, x] = np.median(padded[y : y + k, x : x + k])
    return out.astype(np.uint8)


def _box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """k x k mean filter (works without scipy)."""
    if HAS_SCIPY:
        return ndimage.uniform_filter(arr.astype(np.float64), size=k)
    pad = k // 2
    p = np.pad(arr.astype(np.float64), pad, mode="reflect")
    h, w = arr.shape
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            out += p[dy : dy + h, dx : dx + w]
    return out / (k * k)


def _robust_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * 1.4826 + 1e-6)


def _trimmed_channel_stats(values: np.ndarray, trim_high_frac: float = STAT_TRIM_HIGH_FRAC) -> tuple:
    """Median/MAD/percentile excluding the reddest tail (likely blemishes)."""
    v = np.sort(values)
    n = len(v)
    hi = max(int(n * (1.0 - trim_high_frac)), 1)
    core = v[:hi]
    med = float(np.median(core))
    mad = _robust_mad(core)
    p = float(np.percentile(core, STAT_PERCENTILE_RG))
    return med, mad, p


def _face_skin_color_stats(rgb: np.ndarray, roi: np.ndarray) -> dict:
    """
    RGB / chroma distribution on segmented face skin (for adaptive pimple picking).
    Returns medians, MADs, and percentile thresholds for R-G, R-B, and Cr.
    """
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    rg = r - g
    rb = r - b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    rg_s = rg[roi]
    rb_s = rb[roi]
    cr_s = cr[roi]

    rg_med, rg_mad, p_rg = _trimmed_channel_stats(rg_s)
    rb_med, rb_mad, p_rb = _trimmed_channel_stats(rb_s)
    cr_med, cr_mad, p_cr = _trimmed_channel_stats(cr_s)

    return {
        "r": r,
        "g": g,
        "b": b,
        "rg": rg,
        "rb": rb,
        "cr": cr,
        "med": np.array([rg_med, rb_med, cr_med]),
        "mad": np.array([rg_mad, rb_mad, cr_mad]),
        "p_rg": p_rg,
        "p_rb": p_rb,
        "p_cr": p_cr,
    }


def _face_skin_core(binary_fg: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Eroded interior of the mask — avoids collar, hair fringe, and mask speckles."""
    if HAS_SCIPY:
        return ndimage.binary_erosion(binary_fg, iterations=iterations)
    core = binary_fg.copy()
    for _ in range(iterations):
        pad = 1
        p = np.pad(core.astype(np.uint8), pad, mode="constant")
        h, w = core.shape
        e = np.zeros_like(core)
        for i in range(h):
            for j in range(w):
                e[i, j] = np.all(p[i : i + 3, j : j + 3])
        core = e.astype(bool)
    return core


def detect_pink_pimples_rgb(
    rgb: np.ndarray,
    binary_fg: np.ndarray,
    face_area: np.ndarray,
    eye_regions: np.ndarray,
    nose_regions: np.ndarray,
) -> np.ndarray:
    """
    Pick red/pink dots from RGB using the face-skin color distribution, then blob-filter.
    1) Fit robust stats (median + MAD) on R-G, R-B, Cr inside segmented face skin.
    2) Flag pixels redder than skin (global z) and redder than local neighborhood.
    3) Keep small connected components as pimple mask (before median filter).
    """
    mouth = _mouth_exclusion_mask(binary_fg)
    ears = _ear_exclusion_mask(binary_fg)
    core = _face_skin_core(binary_fg, iterations=4)
    roi = core & face_area & ~eye_regions & ~nose_regions & ~mouth & ~ears
    if not np.any(roi):
        return np.zeros_like(binary_fg, dtype=bool)

    st = _face_skin_color_stats(rgb, roi)
    rg, rb, cr = st["rg"], st["rb"], st["cr"]
    r, g, b = st["r"], st["g"], st["b"]

    # One-sided redness: only above skin median per channel
    feats = np.stack([rg, rb, cr], axis=-1)
    delta = np.maximum(feats - st["med"], 0.0)
    z_global = np.max(delta / st["mad"], axis=2)

    # Local: redder than 15x15 neighborhood on R-G
    local_rg = _box_mean(rg, STAT_LOCAL_WIN)
    drg = rg - local_rg
    drg_roi = drg[roi]
    drg_sorted = np.sort(drg_roi)
    hi = max(int(len(drg_sorted) * 0.94), 1)
    mad_drg = _robust_mad(drg_sorted[:hi])
    z_local = np.maximum(drg, 0.0) / mad_drg
    p_drg = float(np.percentile(drg_sorted[:hi], STAT_PERCENTILE_DRG))

    # Lip-like: high R-G but G also high vs skin (not a spot)
    med_r = float(np.median(r[roi]))
    med_g = float(np.median(g[roi]))
    med_rb = float(np.median(rb[roi]))
    not_lip = ~((g > med_g + 10) & (rb < med_rb + 14))
    not_beard = ~(((r < 155) & (g < 115) & (b < 105)) | ((r < 170) & (g < 130) & (b < 110) & (rg < 38)))

    chroma_count = (
        (rg >= st["p_rg"]).astype(np.uint8)
        + (rb >= st["p_rb"]).astype(np.uint8)
        + (cr >= st["p_cr"]).astype(np.uint8)
    )
    red_excess = r - np.minimum(g, b)
    p_red_excess = float(np.percentile(red_excess[roi], max(STAT_PERCENTILE_RG - 2, 85)))
    p_deep_re = float(np.percentile(red_excess[roi], STAT_DEEP_RED_EXCESS_PCT))
    p_deep_rg = float(np.percentile(rg[roi], STAT_DEEP_RG_PCT))

    stat_hit = (
        roi
        & not_lip
        & not_beard
        & (
            ((z_global >= STAT_GLOBAL_Z) & (z_local >= STAT_LOCAL_Z) & (drg >= STAT_LOCAL_DRG_MIN))
            | ((z_global >= STAT_GLOBAL_Z + 0.35) & (drg >= STAT_LOCAL_DRG_MIN))
            | (
                (rg >= st["p_rg"])
                & (rb >= st["p_rb"])
                & (drg >= p_drg)
                & (z_local >= STAT_LOCAL_Z - 0.4)
            )
            | (
                (chroma_count >= STAT_CHROMA_CHANNELS_MIN)
                & (z_global >= 2.3)
                & (z_local >= STAT_LOCAL_Z)
                & (drg >= STAT_LOCAL_DRG_MIN)
            )
            | (
                (red_excess >= p_red_excess)
                & (rg >= st["p_rg"] - 3.0)
                & (drg >= 0.3)
            )
            | (
                (red_excess >= p_deep_re)
                & (rg >= p_deep_rg - 5.0)
                & (r >= med_r + 4)
            )
        )
    )

    labeled, num_features = _label_binary(stat_hit)
    blobs = np.zeros_like(binary_fg, dtype=bool)
    for i in range(1, num_features + 1):
        region = labeled == i
        size = int(region.sum())
        if not (RGB_BLOB_MIN <= size <= RGB_BLOB_MAX):
            continue
        if region[core].sum() < 0.6 * size:
            continue
        ys, xs = np.where(region)
        if ys.size == 0:
            continue
        if (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) > 8 * size:
            continue
        blobs[region] = True

    # Saturated red cores (per-component size cap — avoid flooding cheek/forehead)
    deep_red = (
        roi
        & not_lip
        & not_beard
        & (red_excess >= p_deep_re)
        & (rg >= p_deep_rg - 5.0)
    )
    dr_labels, dr_n = _label_binary(deep_red)
    for i in range(1, dr_n + 1):
        region = dr_labels == i
        if int(region.sum()) <= RGB_BLOB_MAX:
            blobs[region] = True

    if np.any(blobs):
        deep_boost = blobs & (red_excess >= p_deep_re)
        if HAS_SCIPY and np.any(deep_boost):
            blobs |= ndimage.binary_dilation(deep_boost, iterations=RGB_DEEP_DILATE)
        if HAS_SCIPY:
            blobs = ndimage.binary_dilation(blobs, iterations=RGB_PIMPLE_MASK_DILATE)
        else:
            for _ in range(RGB_PIMPLE_MASK_DILATE):
                h, w = blobs.shape
                dil = blobs.copy()
                for y, x in zip(*np.where(blobs)):
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dil[ny, nx] = True
                blobs = dil

    blobs &= binary_fg & face_area & ~eye_regions & ~nose_regions & ~mouth & ~ears
    return blobs


def _local_black_dots(binary_fg, face_area, min_size, max_size):
    """
    Black pixels mostly surrounded by white (mask dots even if connected
    to larger background through thin paths).
    """
    h, w = binary_fg.shape
    inv = ~binary_fg
    candidates = np.zeros_like(binary_fg, dtype=bool)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not inv[i, j] or not face_area[i, j]:
                continue
            patch = binary_fg[i - 1 : i + 2, j - 1 : j + 2]
            if patch.sum() >= 5:
                candidates[i, j] = True

    labeled, num_features = _label_binary(candidates)
    dots = np.zeros_like(binary_fg, dtype=bool)
    for i in range(1, num_features + 1):
        region = labeled == i
        size = int(region.sum())
        if min_size <= size <= max_size:
            dots[region] = True
    return dots


class PimpleProcessor:
    """Pimples = mask holes and/or pink RGB spots; eyes excluded by hole size."""

    def detect_pimple_mask(
        self,
        face_mask: np.ndarray,
        eye_regions: np.ndarray = None,
        nose_regions: np.ndarray = None,
        rgb: np.ndarray = None,
    ) -> np.ndarray:
        """
        Mask holes (<= PIMPLE_HOLE_MAX) plus optional pink blemishes on RGB
        inside the segmented face. Eyes and nose holes are excluded.
        """
        binary = face_mask > 127
        face_area = _face_area_mask(binary)
        if eye_regions is None:
            eye_regions = extract_eye_regions(binary)
        if nose_regions is None:
            nose_regions = extract_nose_regions(binary)

        pimples = _interior_holes(binary, PIMPLE_HOLE_MIN, PIMPLE_HOLE_MAX)
        pimples |= _small_black_components(binary, face_area, PIMPLE_HOLE_MIN, PIMPLE_HOLE_MAX)
        pimples |= _local_black_dots(binary, face_area, PIMPLE_HOLE_MIN, PIMPLE_HOLE_MAX)
        if rgb is not None:
            pimples |= detect_pink_pimples_rgb(
                rgb, binary, face_area, eye_regions, nose_regions
            )
        pimples &= face_area
        pimples &= ~eye_regions
        pimples &= ~nose_regions
        return pimples

    def apply_median_removal(self, rgb: np.ndarray, pimple_mask: np.ndarray) -> np.ndarray:
        if not np.any(pimple_mask):
            return rgb.copy()

        work_mask = pimple_mask.copy()
        if HAS_SCIPY:
            work_mask = ndimage.binary_dilation(work_mask, iterations=1)

        r0 = rgb[:, :, 0].astype(np.float32)
        g0 = rgb[:, :, 1].astype(np.float32)
        b0 = rgb[:, :, 2].astype(np.float32)
        red_excess0 = _red_excess(r0, g0, b0)
        deep_mask = work_mask.copy()
        if np.any(work_mask):
            deep_thresh = float(np.percentile(red_excess0[work_mask], 52))
            deep_mask = work_mask & (red_excess0 >= deep_thresh)

        out = rgb.copy().astype(np.float32)
        for c in range(3):
            channel = rgb[:, :, c].astype(np.float32)
            for _ in range(MEDIAN_PASSES):
                med = _median_blur_channel(channel.astype(np.uint8), MEDIAN_FILTER_SIZE)
                channel = np.where(work_mask, med.astype(np.float32), channel)
            for _ in range(MEDIAN_DEEP_PASSES):
                med_deep = _median_blur_channel(channel.astype(np.uint8), MEDIAN_DEEP_KERNEL)
                channel = np.where(deep_mask, med_deep.astype(np.float32), channel)
            out[:, :, c] = channel

        return np.clip(out, 0, 255).astype(np.uint8)

    def fill_pimple_holes_in_mask(self, face_mask: np.ndarray, pimple_mask: np.ndarray) -> np.ndarray:
        result = (face_mask > 127) | pimple_mask
        return (result * 255).astype(np.uint8)


class MorphologyRefiner:
    """
    Build a solid face-region mask for Member 3:
    remove all small dots/holes inside the face area.
    """

    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
        self.se = np.zeros((kernel_size, kernel_size), dtype=bool)
        center = kernel_size // 2
        self.se[center, :] = True
        self.se[:, center] = True

    def erode(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        if HAS_SCIPY:
            eroded = ndimage.binary_erosion(binary, structure=self.se)
        else:
            eroded = self._morph(binary, "erode")
        return (eroded * 255).astype(np.uint8)

    def dilate(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        if HAS_SCIPY:
            dilated = ndimage.binary_dilation(binary, structure=self.se)
        else:
            dilated = self._morph(binary, "dilate")
        return (dilated * 255).astype(np.uint8)

    def _morph(self, binary, op):
        ks = self.kernel_size
        pad = ks // 2
        padded = np.pad(binary.astype(np.uint8), ((pad, pad), (pad, pad)), mode="constant")
        h, w = binary.shape
        out = np.zeros_like(binary)
        for i in range(h):
            for j in range(w):
                region = padded[i : i + ks, j : j + ks]
                if op == "erode":
                    out[i, j] = np.all(region[self.se])
                else:
                    out[i, j] = np.any(region[self.se])
        return out

    def opening(self, mask: np.ndarray) -> np.ndarray:
        return self.dilate(self.erode(mask))

    def closing(self, mask: np.ndarray) -> np.ndarray:
        return self.erode(self.dilate(mask))

    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        labeled, num_features = _label_binary(binary)
        if num_features <= 1:
            return mask
        sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
        largest = int(np.argmax(sizes)) + 1
        return ((labeled == largest) * 255).astype(np.uint8)

    def fill_all_holes(self, mask: np.ndarray) -> np.ndarray:
        binary = mask > 127
        if HAS_SCIPY:
            filled = ndimage.binary_fill_holes(binary)
        else:
            inv = ~binary
            h, w = binary.shape
            visited = np.zeros((h, w), dtype=bool)
            stack = []
            for x in range(w):
                if inv[0, x]:
                    stack.append((0, x))
                if inv[h - 1, x]:
                    stack.append((h - 1, x))
            for y in range(h):
                if inv[y, 0]:
                    stack.append((y, 0))
                if inv[y, w - 1]:
                    stack.append((y, w - 1))
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w or visited[y, x] or not inv[y, x]:
                    continue
                visited[y, x] = True
                stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
            filled = binary | (inv & ~visited)
        return (filled * 255).astype(np.uint8)

    def fill_face_black_dots(self, mask: np.ndarray) -> np.ndarray:
        """Fill all small black dots in the face area (including any left after pimples)."""
        binary = mask > 127
        face_area = _face_area_mask(binary)
        dots = _local_black_dots(binary, face_area, PIMPLE_HOLE_MIN, PIMPLE_HOLE_MAX)
        dots |= _small_black_components(binary, face_area, PIMPLE_HOLE_MIN, PIMPLE_HOLE_MAX)
        filled = binary | dots
        return (filled * 255).astype(np.uint8)

    def refine_face_region(self, mask: np.ndarray, eye_regions: np.ndarray = None) -> np.ndarray:
        """
        Solid face-region mask for Member 3: fill all holes and dots,
        including eyes (eye_regions from original mask are filled in here).
        """
        step1 = self.keep_largest_component(mask)
        step2 = self.fill_face_black_dots(step1)
        if eye_regions is not None:
            step2 = (((step2 > 127) | eye_regions) * 255).astype(np.uint8)
        step3 = self.fill_all_holes(step2)
        for _ in range(5):
            step3 = self.closing(step3)
        step4 = self.opening(step3)
        step5 = self.fill_all_holes(step4)
        return step5


def list_member1_mask_files(input_dir: str = INPUT_DIR) -> list:
    """Face masks from segmentation.py ({name}_mask.png, not mask_skin or legacy v4)."""
    if not os.path.isdir(input_dir):
        return []
    return sorted(
        f
        for f in os.listdir(input_dir)
        if f.endswith(MEMBER1_MASK_SUFFIX) and not f.endswith(MEMBER1_MASK_SKIN_SUFFIX)
    )


def base_name_from_mask_file(mask_file: str) -> str:
    if mask_file.endswith(MEMBER1_MASK_SUFFIX):
        return mask_file[: -len(MEMBER1_MASK_SUFFIX)]
    # Legacy fallback if old segmentation outputs are still present
    if mask_file.endswith("_mask_v4.png"):
        return mask_file[: -len("_mask_v4.png")]
    raise ValueError(f"Not a Member 1 mask filename: {mask_file}")


def member1_mask_path(input_dir: str, base_name: str) -> str:
    """Path to Member 1 face mask; prefers {base}_mask.png over legacy _mask_v4.png."""
    primary = os.path.join(input_dir, base_name + MEMBER1_MASK_SUFFIX)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(input_dir, base_name + "_mask_v4.png")
    if os.path.exists(legacy):
        return legacy
    return primary


def read_mask(path):
    mask = plt.imread(path)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def read_rgb(path):
    img = plt.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def find_original_image(base_name):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        path = os.path.join(ORIGINAL_DIR, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def process_one(base_name, mask_path, refiner, pimple_proc):
    mask = read_mask(mask_path)
    orig_path = find_original_image(base_name)
    if orig_path is None:
        raise FileNotFoundError(f"No original image for {base_name} in {ORIGINAL_DIR}/")

    rgb = read_rgb(orig_path)
    binary = mask > 127
    eye_regions = extract_eye_regions(binary)
    nose_regions = extract_nose_regions(binary)

    pimple_mask = pimple_proc.detect_pimple_mask(mask, eye_regions, nose_regions, rgb=rgb)
    cleaned_rgb = pimple_proc.apply_median_removal(rgb, pimple_mask)
    mask_no_pimples = pimple_proc.fill_pimple_holes_in_mask(mask, pimple_mask)
    refined_mask = refiner.refine_face_region(mask_no_pimples, eye_regions)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pimple_dir = os.path.join(OUTPUT_DIR, "pimple_masks")
    cleaned_dir = os.path.join(OUTPUT_DIR, "cleaned_images")
    os.makedirs(pimple_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)

    plt.imsave(os.path.join(pimple_dir, f"{base_name}_pimple_mask.png"), (pimple_mask * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    cleaned_path = os.path.join(cleaned_dir, f"{base_name}_median_cleaned.png")
    if HAS_CV2:
        cv2.imwrite(cleaned_path, cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR))
    else:
        plt.imsave(cleaned_path, cleaned_rgb)
    plt.imsave(os.path.join(OUTPUT_DIR, f"{base_name}_refined_mask.png"), refined_mask, cmap="gray", vmin=0, vmax=255)

    return {
        "rgb": rgb,
        "mask": mask,
        "eye_regions": eye_regions,
        "nose_regions": nose_regions,
        "pimple_mask": pimple_mask,
        "cleaned_rgb": cleaned_rgb,
        "mask_no_pimples": mask_no_pimples,
        "refined_mask": refined_mask,
    }


def save_visualization(base_name, data):
    viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(data["rgb"])
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(data["mask"], cmap="gray")
    axes[0, 1].set_title("Face mask (Member 1)")
    axes[0, 1].axis("off")

    pm = data["pimple_mask"]
    overlay = data["rgb"].copy()
    overlay[pm] = [255, 0, 0]
    overlay_show = data["rgb"].copy()
    eyes = data.get("eye_regions")
    nose = data.get("nose_regions")
    if eyes is not None:
        overlay_show[eyes] = [0, 128, 255]
    if nose is not None:
        overlay_show[nose] = [0, 200, 0]
    overlay_show[pm] = [255, 0, 0]
    axes[0, 2].imshow(overlay_show)
    axes[0, 2].set_title(f"Red=pimples, blue=eyes, green=nose ({pm.sum()} px)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(data["cleaned_rgb"])
    axes[1, 0].set_title("After median filter")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(data["mask_no_pimples"], cmap="gray")
    axes[1, 1].set_title("Pimple holes filled")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(data["refined_mask"], cmap="gray")
    axes[1, 2].set_title("Solid face mask — eyes filled (Member 3)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(viz_dir, f"{base_name}_pipeline_viz.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main():
    mask_files = list_member1_mask_files(INPUT_DIR)
    if not mask_files:
        print(
            f"No {MEMBER1_MASK_SUFFIX} in {INPUT_DIR}/ — run segmentation.py first."
        )
        return

    refiner = MorphologyRefiner(kernel_size=KERNEL_SIZE)
    pimple_proc = PimpleProcessor()

    print("=" * 70)
    print("Member 2: mask-hole + pink RGB pimples + solid face-region mask")
    print("=" * 70)

    for mask_file in mask_files:
        base = base_name_from_mask_file(mask_file)
        mask_path = os.path.join(INPUT_DIR, mask_file)
        try:
            data = process_one(base, mask_path, refiner, pimple_proc)
            viz = save_visualization(base, data)
            print(f"  {base}: {int(data['pimple_mask'].sum())} pimple px")
            print(f"    refined mask -> {OUTPUT_DIR}/{base}_refined_mask.png")
            print(f"    viz -> {viz}")
        except Exception as e:
            print(f"  {base}: ERROR — {e}")

    print("=" * 70)
    print(f"DONE — outputs in {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
