import os
import glob
import cv2
# 设置输入目标图像目录和GT TXT目录
img_folder = "/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron_x2/train/images"  # 目标图像文件夹
gt_folder = "/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron_x2/train/labels"  # YOLO 格式 GT 文件夹

# 获取所有图像和GT文件
img_files = sorted(glob.glob(os.path.join(img_folder, "*.png")))  # 假设图像是PNG格式
gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.txt")))  # 假设GT是TXT格式

# 检查图像和GT尺寸是否一致
for img_path, gt_path in zip(img_files, gt_files):
    # 读取目标图像
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape  # 获取图像的高度和宽度

    # 读取YOLO格式GT文件
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    # 检查GT文件中的坐标
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"Skipping invalid line in {gt_path}: {line}")
            continue

        # 获取类别ID和坐标点
        try:
            class_id = int(parts[0])  # 类别ID
            coordinates = list(map(float, parts[1:]))  # 转换为浮动数
        except ValueError:
            print(f"Skipping invalid line in {gt_path}: {line}")
            continue

        # YOLO 格式的坐标是相对于图像大小的比例 (x_center, y_center, width, height)，确保其值在 [0, 1] 之间
        if any(coord < 0 or coord > 1 for coord in coordinates):
            print(f"Warning: Coordinates out of range in {gt_path}: {line}")
            continue

        # 输出图像和GT文件的尺寸
        if img_h != int(coordinates[2] * img_h) or img_w != int(coordinates[3] * img_w):
            print(f"Warning: Image size and GT size mismatch! {img_path} - {gt_path}")
        else:
            print(f"Image size and GT size match: {img_path} - {gt_path}")

print("检查完成！")
