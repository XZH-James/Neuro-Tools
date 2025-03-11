import numpy as np
import cv2
import os


def get_image_size(image_path):
    """
    获取图像的大小（宽度和高度）。

    参数:
    - image_path: 图像文件路径。

    返回:
    - 图像大小（宽度，高度）。
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return (image.shape[1], image.shape[0])  # 返回宽度和高度


def txt_to_png(txt_file, output_png, image_size):
    """
    将YOLO格式的txt文件转换为8位PNG图像。

    参数:
    - txt_file: 输入的txt文件路径。
    - output_png: 输出的PNG文件路径。
    - image_size: 输出图像的大小（宽度，高度）。
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # 创建一个全零的掩码图像
    mask = np.zeros(image_size[::-1], dtype=np.uint8)  # 高度在前，宽度在后

    # 解析每行数据
    for line in lines:
        data = line.strip().split()
        # 确保数据长度是偶数，代表坐标对
        if len(data) < 3 or len(data) % 2 != 1:
            print(f"跳过无效行: {line}")
            continue

        # 获取类别
        cls = int(data[0])
        if cls != 0:
            continue  # 只处理类别0

        # 提取坐标对
        coords = list(map(float, data[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * image_size[0])  # 将相对坐标转换为像素坐标
            y = int(coords[i + 1] * image_size[1])
            points.append([x, y])

        # 将点转换为numpy数组并绘制多边形
        points = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, points, 1)  # 填充多边形区域为1

    # 保存掩码图像为PNG格式
    cv2.imwrite(output_png, mask)
    print(f"保存掩码图像: {output_png}")


def batch_convert_txt_to_png(input_txt_folder, input_image_folder, output_folder):
    """
    批量将文件夹中的YOLO格式txt文件转换为PNG格式，PNG大小与对应图像大小一致。

    参数:
    - input_txt_folder: 输入txt文件的文件夹路径。
    - input_image_folder: 对应图像的文件夹路径。
    - output_folder: 输出PNG文件的文件夹路径。
    """
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入txt文件夹中的所有txt文件
    for file_name in os.listdir(input_txt_folder):
        if file_name.endswith('.txt'):
            txt_path = os.path.join(input_txt_folder, file_name)
            image_name = file_name.replace('.txt', '.png')  # 假设图像格式为png
            image_path = os.path.join(input_image_folder, image_name)

            # 检查对应图像是否存在
            if os.path.exists(image_path):
                image_size = get_image_size(image_path)
                png_path = os.path.join(output_folder, image_name)
                txt_to_png(txt_path, png_path, image_size)
                print(f"已转换: {txt_path} -> {png_path}")
            else:
                print(f"未找到对应图像: {image_path}")



# 使用示例
input_txt_folder = r'/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron1/test/labels'  # 输入txt文件的文件夹路径 AVG_ROI1 Images512
input_image_folder = r'/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron1/test/images'  # 输入对应图像的文件夹路径
output_folder = r'/public/xu/deeplearning/ultralytics-yolo11-main/dataset/Neuron1/test/masks'   # 输出png文件的文件夹路径
batch_convert_txt_to_png(input_txt_folder, input_image_folder, output_folder)
