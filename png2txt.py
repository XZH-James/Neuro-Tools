import os
import cv2
import numpy as np


def convert_gt_to_yolo(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            # 读取PNG文件（可以是8位或16位）
            img_path = os.path.join(input_folder, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue

            # 如果图像是16位的，将其转换为8位
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)  # 转换为8-bit图像

            # 检查是否为8-bit图像
            if len(img.shape) != 2 or img.dtype != np.uint8:
                print(f"Error: Image is not in 8-bit grayscale format {img_path}")
                continue

            # 初始化YOLO格式的内容
            yolo_annotations = []

            # 找到所有唯一的类别
            classes = np.unique(img)

            for cls in classes:
                if cls == 0:  # 跳过背景类
                    continue

                # 找到该类的所有轮廓
                mask = (img == cls).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # 初始化边缘坐标
                    edge_coordinates = []

                    for point in contour:
                        x, y = point[0]
                        # 计算归一化后的坐标
                        img_h, img_w = img.shape
                        x_norm = x / img_w
                        y_norm = y / img_h
                        edge_coordinates.append(f"{x_norm:.6f} {y_norm:.6f}")

                    # 保存YOLO格式的注释：类别 索引 边缘坐标
                    annotation = f"0 {' '.join(edge_coordinates)}"
                    yolo_annotations.append(annotation)

            # 写入YOLO格式的GT文件
            base_name = os.path.splitext(file_name)[0].replace("_gtFine_instanceIds", "")
            txt_path = os.path.join(output_folder, base_name + ".txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))


# 输入和输出文件夹路径
input_folder = "/public/xu/deeplearning/yolov8-main/dataset/AVG_ROI1 x2/masks"
output_folder = "/public/xu/deeplearning/yolov8-main/dataset/AVG_ROI1 x2/labels"

convert_gt_to_yolo(input_folder, output_folder)
