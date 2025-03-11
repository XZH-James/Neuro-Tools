import cv2
import os
import glob
import random
import numpy as np

# 设置超分辨率结果目录 & 2x GT 目录
sr_folder = "/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron_x2/val/images"  # 超分辨率图像路径
gt_x2_folder = "/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron_x2/val/masks"  # 放大后的 GT 路径

# 结果保存目录
mismatch_save_folder = "/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron1 "
os.makedirs(mismatch_save_folder, exist_ok=True)

# 读取所有超分 & GT_x2 文件
sr_images = sorted(glob.glob(os.path.join(sr_folder, "*")))
gt_x2_images = sorted(glob.glob(os.path.join(gt_x2_folder, "*")))

# 创建字典，确保文件名匹配
sr_dict = {os.path.basename(f): f for f in sr_images}
gt_x2_dict = {os.path.basename(f): f for f in gt_x2_images}

# 记录尺寸不匹配的图像
mismatch_files = []

# 遍历 SR 目录，检查尺寸
for filename in sr_dict.keys():
    if filename not in gt_x2_dict:
        print(f"Warning: {filename} 在 GT_x2 目录中不存在！")
        continue

    # 读取图像
    img_sr = cv2.imread(sr_dict[filename])
    img_gt_x2 = cv2.imread(gt_x2_dict[filename])

    # 获取尺寸
    h_sr, w_sr, _ = img_sr.shape
    h_gt, w_gt, _ = img_gt_x2.shape

    # 检查尺寸是否匹配
    if (h_sr, w_sr) != (h_gt, w_gt):
        mismatch_files.append(filename)
        print(f"尺寸不匹配: {filename} (SR: {w_sr}x{h_sr}, GT_x2: {w_gt}x{h_gt})")

# **如果没有不匹配的图像，随机选 3 张进行可视化**
if not mismatch_files:
    print("所有超分图像尺寸与 GT_x2 匹配！")
    mismatch_files = random.sample(list(sr_dict.keys()), min(3, len(sr_dict)))

for filename in mismatch_files:
    img_sr = cv2.imread(sr_dict[filename])  # 读取超分图像
    img_gt_x2 = cv2.imread(gt_x2_dict[filename], cv2.IMREAD_GRAYSCALE)  # 读取 GT 并转换为灰度图

    # 确保 GT 只有 0 和 1
    img_gt_x2 = np.where(img_gt_x2 > 0, 1, 0).astype(np.uint8)  # 将大于 0 的像素转换为 1，其他为 0

    # 创建浅红色 mask（BGR: [0, 102, 255]）
    gt_mask = np.zeros_like(img_sr, dtype=np.uint8)
    gt_mask[:, :, 0] = 0  # B
    gt_mask[:, :, 1] = 0  # G
    gt_mask[:, :, 2] = 255  # R
    gt_mask = gt_mask * img_gt_x2[:, :, np.newaxis]  # 只在目标区域应用颜色

    # 叠加红色 GT 到超分图像
    blended = cv2.addWeighted(img_sr, 0.7, gt_mask, 0.3, 0)

    # 保存可视化结果
    save_path = os.path.join(mismatch_save_folder, f"mismatch_{filename}")
    cv2.imwrite(save_path, blended)
    print(f"✅ 叠加 GT 的可视化图已保存: {save_path}")

print("Finished")
