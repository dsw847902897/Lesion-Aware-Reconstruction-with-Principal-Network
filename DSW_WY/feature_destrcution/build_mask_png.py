from PIL import Image, ImageDraw

img_path='try_cases/test_wrong.png'

x=(90,314,148,360)   #(xmin,ymin,xmax,ymax)

image = Image.open(img_path)
w,h = image.size

# 创建一个带有RGBA模式（包含透明度）的新图像
img = Image.new('RGBA', (w,h), (0, 0, 0, 0))
# 用图像作为背景初始化绘图上下文
draw = ImageDraw.Draw(img)
# 绘制一个完全不透明的白色矩形
draw.rectangle([(x[0]-2,x[1]-2),(x[2]+2,x[3]+2)], fill=(255, 255, 255, 255))
# 将图像保存到文件
mask_path=img_path.split('.')[0]+'_mask.png'
img.save(mask_path)

# 创建一个可以在给定图像上绘图的对象
draw1 = ImageDraw.Draw(image)
# 设置要变白的矩形区域，格式为[左上角x, 左上角y, 右下角x, 右下角y]
rectangle_area = [x[0]-2, x[1]-2, x[2]+2, x[3]+2]
# 用白色填充指定的矩形区域
draw1.rectangle(rectangle_area, fill='white')
# 保存修改后的图像
input_path=img_path.split('.')[0]+'_input.png'
image.save(input_path)