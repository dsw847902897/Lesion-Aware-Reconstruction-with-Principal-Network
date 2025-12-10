import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

def random_shift_box(gt_box, iou_range=(0.5, 0.7), max_attempts=10000):
    x1, y1, x2, y2 = gt_box
    gt_area = (x2 - x1) * (y2 - y1)

    for _ in range(max_attempts):
        delta_x1 = np.random.uniform(-1, 1) * (x2 - x1)
        delta_y1 = np.random.uniform(-1, 1) * (y2 - y1)
        delta_x2 = np.random.uniform(-1, 1) * (x2 - x1)
        delta_y2 = np.random.uniform(-1, 1) * (y2 - y1)

        new_box = [
            x1 + delta_x1,
            y1 + delta_y1,
            x2 + delta_x2,
            y2 + delta_y2
        ]

        # 计算 IoU
        iou = calculate_iou(gt_box, new_box)

        # 检查 IoU 是否在目标范围内
        if iou_range[0] <= iou <= iou_range[1]:
            return new_box

    return None  # 如果没有找到符合条件的框，返回 None


# 计算 IoU 的函数
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union


# 解析 XML 文件
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bndbox = root.find(".//bndbox")
    xmin = float(bndbox.find("xmin").text)
    ymin = float(bndbox.find("ymin").text)
    xmax = float(bndbox.find("xmax").text)
    ymax = float(bndbox.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


# 将多个框坐标保存到 TXT 文件
def save_to_txt(boxes, file_path):
    with open(file_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")


if __name__ == "__main__":
    xml_dir = "/home/wy/dataset/topicData/MainProject/Annotations/"  # 替换为你的 XML 文件路径
    output_txt_dir = "/home/wy/py_doc/MainPart2/IouBox_generate/iou_txt"  # 输出的 TXT 文件路径
    num_boxes = 10  # 需要生成的框的数量
    iou_range = (0.1, 0.2)  # IoU 范围

    for case in tqdm(os.listdir(xml_dir)):
        xml_path=os.path.join(xml_dir,case)
        output_txt_path=os.path.join(output_txt_dir,case.split(".")[0]+".txt")

        # 解析 XML 文件，获取 Ground Truth Box
        gt_box = parse_xml(xml_path)

        # 生成多个检测框
        generated_boxes = []
        for _ in range(num_boxes):
            new_box = random_shift_box(gt_box, iou_range=iou_range)
            if new_box:
                generated_boxes.append(new_box)

        if generated_boxes:
            print(f"Generated {len(generated_boxes)} boxes:")
            for box in generated_boxes:
                print(box)
            # 将生成的坐标保存到 TXT 文件
            save_to_txt(generated_boxes, output_txt_path)
            print(f"Box coordinates saved to {output_txt_path}")
        else:
            print("Failed to generate any boxes within the specified IoU range.")