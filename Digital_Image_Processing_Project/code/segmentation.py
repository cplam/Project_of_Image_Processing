

import os
import numpy as np
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.path.dirname(__file__))
# INPUT_PATH = os.path.join(parent_dir, "dataset")
# OUTPUT_DIR = "./output/output_member1"
INPUT_PATH = os.path.join(parent_dir, "input_image")
# OUTPUT_DIR = "./test_output"
OUTPUT_DIR = "./output/output_member1"


CB_LOW, CB_HIGH = 77, 127
CR_LOW, CR_HIGH = 133, 173


def rgb_to_ycbcr(rgb):
    """ Y = 0.299R + 0.587G + 0.114B
        Cb = -0.169R - 0.331G + 0.500B + 128
        Cr = 0.500R - 0.419G - 0.081B + 128
    """
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def otsu_threshold(gray):
    """Histogram-based Global Method"""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()

    best_t, max_var = 0, 0
    intensities = np.arange(256)

    for t in range(1, 256):
        w0, w1 = hist[:t].sum(), hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (intensities[:t] * hist[:t]).sum() / w0
        mu1 = (intensities[t:] * hist[t:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > max_var:
            max_var, best_t = var, t
    return best_t


def ycbcr_skin_mask(ycbcr, cb_low=CB_LOW, cb_high=CB_HIGH,
                    cr_low=CR_LOW, cr_high=CR_HIGH):
    cb, cr = ycbcr[:, :, 1], ycbcr[:, :, 2]
    mask = ((cb >= cb_low) & (cb <= cb_high) &
            (cr >= cr_low) & (cr <= cr_high))
    return mask.astype(np.uint8) * 255


def segment_v4(rgb):
    ycbcr = rgb_to_ycbcr(rgb)
    y = ycbcr[:, :, 0]
    mask_skin = ycbcr_skin_mask(ycbcr)

    skin_ratio = np.mean(mask_skin > 0)

    if skin_ratio < 0.2:
        t = otsu_threshold(y)
        mask = (y > t).astype(np.uint8) * 255
    else:
        y_in_mask_skin = np.where(mask_skin > 0, y, 0)
        t = otsu_threshold(y_in_mask_skin)
        mask = ((y > t) & (mask_skin > 0)).astype(np.uint8) * 255

    # mask_final = refine_mask(mask)

    return mask, {
        "otsu_t": t,
        "ycbcr": ycbcr,
        "mask_skin": mask_skin,
    }


def read_rgb(path):
    """Read image as uint8 RGB numpy array."""
    img = plt.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def save_image(array, path):
    """Save numpy array as image file."""
    if array.ndim == 2:   # grayscale / mask
        plt.imsave(path, array, cmap='gray', vmin=0, vmax=255)
    else:                  # RGB
        plt.imsave(path, array)


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(INPUT_PATH)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    for file in files:
        rgb = read_rgb(os.path.join(INPUT_PATH, file))
        mask, info = segment_v4(rgb)
        save_image(mask, os.path.join(OUTPUT_DIR, f"{file.split('.')[0]}_mask.png"))
        save_image(info["mask_skin"], os.path.join(OUTPUT_DIR, f"{file.split('.')[0]}_mask_skin.png"))
    
    print("=" * 50)
    print(f"DONE! Output saved to {OUTPUT_DIR}/")
    print("Ready for Member 2")
    print("=" * 50)


if __name__ == "__main__":
    main()
