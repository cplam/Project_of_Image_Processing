#!/usr/bin/env python3
"""
noise_adder.py - 为文件夹中的所有图像添加多种噪声：
    - Gaussian noise (加性高斯噪声)
    - Uniform white noise (加性均匀白噪声)
    - Poisson noise (泊松噪声)
    - Speckle noise (乘性斑点噪声)
    - Salt & Pepper noise (椒盐噪声)

用法示例:
    # 只添加高斯噪声和椒盐噪声（默认行为，兼容旧版）
    python noise_adder.py --input_dir ./images --output_dir ./noised \
        --gaussian_var 25.0 --salt_prob 0.05

    # 添加均匀白噪声和泊松噪声
    python noise_adder.py --input_dir ./images --output_dir ./noised \
        --uniform --uniform_range 30 --poisson

    # 添加所有噪声类型
    python noise_adder.py --input_dir ./images --output_dir ./noised --all

参数说明:
    --input_dir         输入图像文件夹路径 (默认: 当前目录)
    --output_dir        输出文件夹路径 (默认: ./noised_output)

    --all               添加所有支持的噪声类型（覆盖下面单项开关）

    --gaussian          添加高斯噪声 (默认: True)
    --gaussian_var      高斯噪声方差 (默认: 25.0, 适合 0-255 范围)
    --gaussian_prob     高斯噪声应用概率 (默认: 1.0)

    --uniform           添加均匀白噪声 (默认: False)
    --uniform_range     均匀噪声的半宽，实际范围 [-range, +range] (默认: 30)

    --poisson           添加泊松噪声 (默认: False)
    --poisson_scale     泊松噪声缩放因子，值越大噪声越强 (默认: 1.0)

    --speckle           添加乘性斑点噪声 (默认: False)
    --speckle_var       斑点噪声方差 (默认: 0.04)

    --saltpepper        添加椒盐噪声 (默认: True)
    --salt_prob         椒盐噪声总概率 (默认: 0.05)
"""

import os
import argparse
import numpy as np
from PIL import Image

# ========== 噪声生成函数 ==========
def add_gaussian_noise(img_array, variance, prob=1.0):
    """加性高斯噪声（零均值）"""
    img_float = img_array.astype(np.float32)
    noise = np.random.normal(0, np.sqrt(variance), img_float.shape)
    if prob < 1.0:
        mask = np.random.random(img_float.shape) < prob
        img_float = img_float + noise * mask
    else:
        img_float = img_float + noise
    return np.clip(img_float, 0, 255).astype(np.uint8)

def add_uniform_white_noise(img_array, half_range):
    """加性均匀分布白噪声，范围 [-half_range, +half_range]"""
    img_float = img_array.astype(np.float32)
    noise = np.random.uniform(-half_range, half_range, img_float.shape)
    img_float = img_float + noise
    return np.clip(img_float, 0, 255).astype(np.uint8)

def add_poisson_noise(img_array, scale=1.0):
    """
    泊松噪声：模拟光子计数噪声。
    原理：对于每个像素值 I，生成服从泊松分布的随机数，均值为 I/scale，再乘回 scale。
    当 scale 越大，噪声相对越弱。
    """
    img_float = img_array.astype(np.float32)
    # 避免负值或零导致泊松分布出错
    img_scaled = np.maximum(img_float, 0) / scale
    noisy_scaled = np.random.poisson(img_scaled)
    noisy = noisy_scaled * scale
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(img_array, variance):
    """
    乘性斑点噪声：output = I + I * N(0, var)
    常用于模拟超声或SAR图像噪声。
    """
    img_float = img_array.astype(np.float32)
    noise = np.random.normal(0, np.sqrt(variance), img_float.shape)
    img_noisy = img_float + img_float * noise
    return np.clip(img_noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img_array, prob):
    """
    椒盐噪声：每个像素以 prob/2 概率变为 0，以 prob/2 概率变为 255。
    注意：对于彩色图像，三个通道同时变为相同值。
    """
    noisy = img_array.copy()
    rand = np.random.random(img_array.shape[:2])  # (H, W)
    salt_mask = rand < prob / 2.0
    pepper_mask = (rand >= prob / 2.0) & (rand < prob)
    for c in range(img_array.shape[2]):
        noisy[..., c][salt_mask] = 255
        noisy[..., c][pepper_mask] = 0
    return noisy

# ========== 主处理逻辑 ==========
def process_images(input_dir, output_dir, args):
    # 支持的图像扩展名
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print(f"在 {input_dir} 中没有找到图像文件。")
        return
    
    print(f"找到 {len(files)} 个图像文件，开始处理...")
    
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img_array = np.array(img)  # (H, W, 3)
            base_name = os.path.splitext(filename)[0]
            
            # 决定要添加哪些噪声
            add_list = []
            if args.all:
                add_list = ['gaussian', 'uniform', 'poisson', 'speckle', 'saltpepper']
            else:
                if args.gaussian:
                    add_list.append('gaussian')
                if args.uniform:
                    add_list.append('uniform')
                if args.poisson:
                    add_list.append('poisson')
                if args.speckle:
                    add_list.append('speckle')
                if args.saltpepper:
                    add_list.append('saltpepper')
            
            # 依次添加并保存
            for noise_type in add_list:
                if noise_type == 'gaussian':
                    noisy = add_gaussian_noise(img_array, args.gaussian_var, args.gaussian_prob)
                    suffix = f"_gaussian_var{args.gaussian_var}"
                elif noise_type == 'uniform':
                    noisy = add_uniform_white_noise(img_array, args.uniform_range)
                    suffix = f"_uniform_range{args.uniform_range}"
                elif noise_type == 'poisson':
                    noisy = add_poisson_noise(img_array, args.poisson_scale)
                    suffix = f"_poisson_scale{args.poisson_scale}"
                elif noise_type == 'speckle':
                    noisy = add_speckle_noise(img_array, args.speckle_var)
                    suffix = f"_speckle_var{args.speckle_var}"
                elif noise_type == 'saltpepper':
                    noisy = add_salt_pepper_noise(img_array, args.salt_prob)
                    suffix = f"_saltpepper_prob{args.salt_prob}"
                else:
                    continue
                
                out_name = f"{base_name}{suffix}.png"
                out_path = os.path.join(output_dir, out_name)
                Image.fromarray(noisy).save(out_path)
                print(f"已生成: {out_name}")
        
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="为图像添加多种噪声")
    parser.add_argument('--input_dir', type=str, default='.',
                        help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./noised_output',
                        help='输出文件夹路径')
    
    # 噪声开关
    parser.add_argument('--all', action='store_true', help='添加所有支持的噪声类型')
    parser.add_argument('--gaussian', action='store_true', default=True,
                        help='添加高斯噪声 (默认开启)')
    parser.add_argument('--uniform', action='store_true', default=True,
                        help='添加均匀白噪声 (默认开启)')
    parser.add_argument('--poisson', action='store_true', default=True,
                        help='添加泊松噪声 (默认开启)')
    parser.add_argument('--speckle', action='store_true', default=True,
                        help='添加乘性斑点噪声 (默认开启)')
    parser.add_argument('--saltpepper', action='store_true', default=True,
                        help='添加椒盐噪声 (默认开启)')
    
    # 噪声参数
    parser.add_argument('--gaussian_var', type=float, default=1.5,
                        help='高斯噪声方差')
    parser.add_argument('--gaussian_prob', type=float, default=3.0,
                        help='高斯噪声应用概率')
    parser.add_argument('--uniform_range', type=float, default=30.0,
                        help='均匀噪声的半宽，实际范围 [-range, +range]')
    parser.add_argument('--poisson_scale', type=float, default=3.0,
                        help='泊松噪声缩放因子，值越大噪声越弱')
    parser.add_argument('--speckle_var', type=float, default=0.04,
                        help='斑点噪声方差')
    parser.add_argument('--salt_prob', type=float, default=0.05,
                        help='椒盐噪声总概率')
    
    args = parser.parse_args()
    
    # 参数合法性检查
    if args.gaussian_var < 0:
        args.gaussian_var = 0
    if not (0 <= args.gaussian_prob <= 1):
        args.gaussian_prob = max(0, min(1, args.gaussian_prob))
    if args.uniform_range < 0:
        args.uniform_range = 0
    if args.poisson_scale <= 0:
        args.poisson_scale = 0.01
    if args.speckle_var < 0:
        args.speckle_var = 0
    if not (0 <= args.salt_prob <= 1):
        args.salt_prob = max(0, min(1, args.salt_prob))
    
    process_images(args.input_dir, args.output_dir, args)
    print("处理完成！")

if __name__ == '__main__':
    main()