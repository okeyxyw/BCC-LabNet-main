import cv2
import numpy as np
import json
import os
from pathlib import Path
import random


def simulate_low_light(image, dark_factor, noise_level=5):
    """
    模拟低光照条件
    :param image: 输入图像
    :param dark_factor: 暗化因子 (0-1)
    :param noise_level: 噪声水平
    :return: 低光照图像
    """
    # 降低亮度
    dark_image = image * dark_factor
    dark_image = np.clip(dark_image, 0, 255).astype(np.uint8)

    # 添加轻微噪声模拟低光传感器噪声
    noise = np.random.normal(0, noise_level, dark_image.shape).astype(np.uint8)
    noisy_image = cv2.add(dark_image, noise)

    return noisy_image


def batch_simulate_low_light(input_dir, output_dir, dark_factor_range=(0.1, 0.3)):
    """
    批量生成低光照数据集，使用更暗的暗化因子范围
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 存储暗化因子对应关系
    dark_factor_mapping = {}
    processed_count = 0

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"无法读取图像: {img_file.name}")
                continue

            # 为每张图片生成随机暗化因子（使用更暗的范围）
            dark_factor = random.uniform(dark_factor_range[0], dark_factor_range[1])

            # 生成低光照图像
            low_light_img = simulate_low_light(image, dark_factor=dark_factor)

            # 保持原文件名
            output_filepath = output_path / img_file.name

            # 保存图像
            cv2.imwrite(str(output_filepath), low_light_img)

            # 记录暗化因子
            dark_factor_mapping[img_file.name] = float(dark_factor)
            processed_count += 1
            print(f"生成: {img_file.name}, 暗化因子: {dark_factor:.3f}")

    # 保存暗化因子映射到JSON文件
    mapping_file = output_path / "dark_factor_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(dark_factor_mapping, f, indent=4, ensure_ascii=False)

    # 同时保存为易读的文本文件
    txt_file = output_path / "dark_factor_mapping.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("文件名\t\t暗化因子\n")
        f.write("-" * 40 + "\n")
        for filename, factor in dark_factor_mapping.items():
            f.write(f"{filename}\t{factor:.4f}\n")

    print(f"\n完成！共处理 {processed_count} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"暗化因子映射文件: {mapping_file}")
    print(f"文本格式映射文件: {txt_file}")

    # 显示暗化因子统计信息
    factors = list(dark_factor_mapping.values())
    print(f"\n暗化因子统计:")
    print(f"最小值: {min(factors):.3f}")
    print(f"最大值: {max(factors):.3f}")
    print(f"平均值: {np.mean(factors):.3f}")

    return dark_factor_mapping


# 更真实的版本，调整为更暗但不增加噪声
def realistic_low_light_simulation(image, dark_factor, illumination_variance=0.5, noise_std=5):
    """
    更真实的低光照模拟，调整为更暗的效果但不增加噪声
    """
    # 转换为浮点计算
    img_float = image.astype(np.float32) / 255.0

    # 生成非均匀光照图
    h, w = img_float.shape[:2]
    y, x = np.ogrid[:h, :w]

    # 随机生成光照中心点
    center_x = random.uniform(0.3 * w, 0.7 * w)
    center_y = random.uniform(0.3 * h, 0.7 * h)

    # 随机生成光照衰减范围
    sigma_x = random.uniform(0.15 * w, 0.35 * w)  # 减小衰减范围，使暗区更明显
    sigma_y = random.uniform(0.15 * h, 0.35 * h)

    # 创建非均匀光照图
    illumination_map = np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
                                (y - center_y) ** 2 / (2 * sigma_y ** 2)))

    # 调整光照图强度变化（增大变化幅度，制造更暗区域）
    illumination_map = illumination_map * (1 - illumination_variance) + illumination_variance

    # 应用基础暗化因子和非均匀光照
    if len(img_float.shape) == 3:
        illumination_map = illumination_map[:, :, np.newaxis]

    # 使用更低的基础亮度
    dark_image = img_float * dark_factor * illumination_map

    # 保持原有噪声水平
    noise_intensity = noise_std / 255.0
    noise = np.random.normal(0, noise_intensity, dark_image.shape)
    noisy_dark_image = dark_image + noise

    noisy_dark_image = np.clip(noisy_dark_image, 0, 1)

    # 转换回8位
    result = (noisy_dark_image * 255).astype(np.uint8)

    return result


def batch_realistic_simulation(input_dir, output_dir, dark_factor_range=(0.15, 0.4)):
    """
    批量真实低光照模拟，使用更暗的参数但不增加噪声
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    dark_factor_mapping = {}
    processed_count = 0

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            # 生成随机暗化因子（使用更暗的范围）
            dark_factor = random.uniform(dark_factor_range[0], dark_factor_range[1])

            # 生成低光照版本
            low_light_img = realistic_low_light_simulation(image, dark_factor)

            # 保持原文件名
            output_filepath = output_path / img_file.name
            cv2.imwrite(str(output_filepath), low_light_img)

            # 记录暗化因子
            dark_factor_mapping[img_file.name] = float(dark_factor)
            processed_count += 1
            print(f"生成: {img_file.name}, 暗化因子: {dark_factor:.3f}")

    # 保存映射文件
    mapping_file = output_path / "dark_factor_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(dark_factor_mapping, f, indent=4, ensure_ascii=False)

    print(f"\n完成！处理了 {processed_count} 张图像")
    print(f"暗化因子映射已保存至: {mapping_file}")

    # 显示统计信息
    factors = list(dark_factor_mapping.values())
    print(f"\n暗化因子统计:")
    print(f"最小值: {min(factors):.3f}")
    print(f"最大值: {max(factors):.3f}")
    print(f"平均值: {np.mean(factors):.3f}")

    return dark_factor_mapping


# 使用方法
if __name__ == "__main__":
    gt_dir = r"data\VisDrone2\test\hgih"
    output_dir = r"data\VisDrone2\test\low"

    print("选择生成模式:")
    print("1. 简单暗化 (较暗)")
    print("2. 真实感模拟 (更暗且真实)")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "2":
        # 使用真实感模拟，暗化因子范围更暗
        mapping = batch_realistic_simulation(gt_dir, output_dir, dark_factor_range=(0.1, 0.3))
    else:
        # 使用简单暗化，暗化因子范围更暗
        mapping = batch_simulate_low_light(gt_dir, output_dir, dark_factor_range=(0.1, 0.3))

    # 打印前几个暗化因子作为示例
    print("\n暗化因子示例:")
    for i, (filename, factor) in enumerate(list(mapping.items())[:5]):
        print(f"{filename}: {factor:.4f}")