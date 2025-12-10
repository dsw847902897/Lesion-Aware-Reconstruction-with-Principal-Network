from PIL import Image
import os

def resize_images(folder_path, size=(256, 256)):
    # 检查输出文件夹是否存在，如果不存在，则创建

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # 检查文件是否是PNG图像
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            # 调整图像大小
            resized_image = image.resize(size, Image.ANTIALIAS)
            
            # 保存调整大小后的图像到输出文件夹
            output_path = os.path.join(folder_path, filename)
            resized_image.save(output_path)


# 设置源文件夹和目标文件夹路径
source_folder = '/home/wy/py_doc/GII/generative_inpainting/data/health'


# 调用函数
resize_images(source_folder)
