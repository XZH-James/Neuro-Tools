import cv2
import os
import glob
import numpy as np

# 设置输入 GT 目录和输出目录
gt_folder = "/public/xu/deeplearning/yolov8-main/dataset/AVG_ROI1 Images512/masks"  # 原始 GT 文件夹
output_folder = "/public/xu/deeplearning/yolov8-main/dataset/AVG_ROI1 x2/masks"  # 放大后的 GT 存放位置

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有 GT 图像
gt_images = sorted(glob.glob(os.path.join(gt_folder, "*")))

# 遍历所有 GT 图像，并进行 2x 放大
for gt_path in gt_images:
    # 读取 GT 图像
    img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 直接以灰度模式读取
    if img_gt is None:
        print(f"Error: Cannot read image {gt_path}")
        continue

    # 获取 GT 图像的原始尺寸
    h, w = img_gt.shape
    new_h, new_w = h * 2, w * 2  # 放大 2 倍

    # 进行 2x 放大（双线性插值）
    img_gt_x2 = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 将像素值转换为8位
    img_gt_x2 = (img_gt_x2 * 255).astype(np.uint8)

    # 生成保存路径
    filename = os.path.basename(gt_path)  # 获取文件名
    save_path = os.path.join(output_folder, filename)  # 目标路径

    # 保存放大后的 GT 图像
    cv2.imwrite(save_path, img_gt_x2)
    print(f"Saved upscaled GT: {save_path}")

print("✅ 所有 GT 图像已放大 2 倍并转换为8bit！")
