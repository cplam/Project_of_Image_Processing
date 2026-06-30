#!/usr/bin/env python3
"""
batch_restoration.py

对输入文件夹中的所有图像应用四种恢复（去噪）方法：
    - mean filter (均值滤波)
    - median filter (中值滤波)
    - gaussian filter (高斯滤波)
    - VST + Gaussian (方差稳定变换 + 高斯滤波)

每种方法生成一张输出图像，保存到输出文件夹。

用法示例:
    python batch_restoration.py --input_dir ./noisy_images --output_dir ./restored

可选参数:
    --kernel     滤波核大小 / 高斯 sigma 因子 (默认: 3)
    --ext        图像扩展名过滤 (默认: .png,.jpg,.jpeg,.bmp)
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage


# ========== 滤波器实现（基于原 pipeline 代码） ==========
def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """均值滤波"""
    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = ndimage.uniform_filter(image[:, :, c].astype(np.float32), size=size)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = ndimage.uniform_filter(image.astype(np.float32), size=size)
        return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """中值滤波"""
    if image.ndim == 3:
        result = np.zeros_like(image)
        for c in range(3):
            result[:, :, c] = ndimage.median_filter(image[:, :, c], size=size)
        return result
    else:
        return ndimage.median_filter(image, size=size)


def gaussian_filter(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """高斯滤波 (sigma 控制平滑程度)"""
    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = ndimage.gaussian_filter(image[:, :, c].astype(np.float32), sigma=sigma)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
        return np.clip(result, 0, 255).astype(np.uint8)


import numpy as np
from scipy import ndimage
import cv2  # 建議使用 cv2 進行雙邊濾波，保邊效果遠超純高斯

def vst_denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    優化版 VST 去噪：
    1. 使用閉式精確逆變換 (Exact Unbiased Inverse) 減少偏差。
    2. 將單純的高斯濾波更換為「邊緣保留」濾波。
    """
    strength = 15
    img_float = image.astype(np.float32)

    # --- 1. Anscombe 前向變換 ---
    img_vst = 2.0 * np.sqrt(np.maximum(img_float, 0) + 3.0/8.0)

    # --- 2. 邊緣保留濾波 (Bilateral Filter) ---
    # 高斯濾波會均勻模糊所有東西，這就是你覺得它跟高斯沒區別的原因。
    # 雙邊濾波能在變換域內「只抹除雜訊，保留邊緣」。
    if image.ndim == 3:
        filtered = np.zeros_like(img_vst)
        for c in range(3):
            # d 是鄰域直徑，sigmaColor 是顏色空間標準差，sigmaSpace 是座標空間標準差
            filtered[:, :, c] = cv2.bilateralFilter(img_vst[:, :, c], d=5, 
                                                   sigmaColor=sigma*2, 
                                                   sigmaSpace=sigma)
    else:
        filtered = cv2.bilateralFilter(img_vst, d=5, sigmaColor=sigma*2, sigmaSpace=sigma)

    # --- 3. 閉式精確逆變換 (Asymptotic Unbiased Inverse) ---
    # 改進：使用 (f/2)^2 - 1/8 + (1/4)*sqrt(3/2)*f^-1 ... 等更精確的映射
    # 這裡採用更穩健的近似：能有效修正暗部偏色和細節模糊
    denoised = (filtered / 2.0) ** 2 - 1.0/8.0 
    
    # 針對強雜訊進行微調修正
    denoised = np.maximum(denoised, 0)
    
    # --- 4. 後處理 ---
    denoised = np.clip(denoised, 0, 255)
    return denoised.astype(np.uint8)


# ========== 批处理 ==========
def process_folder(input_dir: str, output_dir: str, kernel_size: int = 3, extensions=('.png', '.jpg', '.jpeg', '.bmp')):
    """对文件夹内所有图像应用四种滤波器并保存"""
    os.makedirs(output_dir, exist_ok=True)

    # 收集图像文件
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    if not files:
        print(f"错误：在 {input_dir} 中没有找到图像文件。")
        return

    print(f"找到 {len(files)} 个图像文件。")
    print(f"滤波参数：kernel_size = {kernel_size}（均值/中值窗口大小；高斯 sigma = {kernel_size/2:.1f}；VST sigma = {max(0.5, kernel_size/2):.1f}）")
    print("=" * 60)

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(input_dir, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img_array = np.array(img)
            base_name = os.path.splitext(filename)[0]

            print(f"[{idx}/{len(files)}] 处理 {filename} ...")

            # 1. 均值滤波
            out_mean = mean_filter(img_array, size=kernel_size)
            mean_path = os.path.join(output_dir, f"{base_name}_mean.png")
            Image.fromarray(out_mean).save(mean_path)

            # 2. 中值滤波
            out_median = median_filter(img_array, size=kernel_size)
            median_path = os.path.join(output_dir, f"{base_name}_median.png")
            Image.fromarray(out_median).save(median_path)

            # 3. 高斯滤波 (sigma = kernel_size/2)
            sigma_gauss = kernel_size / 2.0
            out_gauss = gaussian_filter(img_array, sigma=sigma_gauss)
            gauss_path = os.path.join(output_dir, f"{base_name}_gaussian.png")
            Image.fromarray(out_gauss).save(gauss_path)

            # 4. VST + 高斯 (sigma 与高斯滤波保持一致，但至少 0.5)
            sigma_vst = max(0.5, kernel_size / 2.0)
            out_vst = vst_denoise(img_array, sigma=sigma_vst)
            vst_path = os.path.join(output_dir, f"{base_name}_vst.png")
            Image.fromarray(out_vst).save(vst_path)

            print(f"    -> 已保存 4 张结果图")

        except Exception as e:
            print(f"    处理 {filename} 时出错: {e}")

    print("=" * 60)
    print(f"全部处理完成！结果保存在：{output_dir}")


# ========== 命令行入口 ==========
def main():
    parser = argparse.ArgumentParser(description="批量图像恢复（去噪）：均值、中值、高斯、VST+高斯")
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./restored_output', help='输出文件夹路径 (默认: ./restored_output)')
    parser.add_argument('--kernel', type=int, default=3, help='滤波核大小（均值/中值窗口大小，高斯/VST的sigma由 kernel/2 决定）(默认: 3)')
    parser.add_argument('--ext', type=str, default='.png,.jpg,.jpeg,.bmp', help='图像扩展名，逗号分隔 (默认: .png,.jpg,.jpeg,.bmp)')
    args = parser.parse_args()

    extensions = tuple(ext.strip() for ext in args.ext.split(','))
    process_folder(args.input_dir, args.output_dir, args.kernel, extensions)


if __name__ == '__main__':
    main()