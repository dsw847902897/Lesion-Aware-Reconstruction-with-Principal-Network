from PIL import Image, ImageDraw
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


v=0.15
def out_mask_input(raw_img_path,x,save_dir):
    raw_img=Image.open(raw_img_path)
    w,h=raw_img.size

    mask=Image.new('RGBA', (w,h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    xv=0.15*(x[2]-x[0])
    yv=0.15*(x[3]-x[1])
    draw.rectangle([(x[0]-xv,x[1]-yv),(x[2]+xv,x[3]+yv)], fill=(255, 255, 255, 255))
    mask_path=os.path.join(save_dir,'mask',os.path.basename(raw_img_path).split('.')[0]+'_mask.png')
    
    mask.save(mask_path)

    draw1=ImageDraw.Draw(raw_img)

    rectangle_area=[x[0]-xv, x[1]-yv, x[2]+xv, x[3]+yv]
    draw1.rectangle(rectangle_area,fill='white')
    input_path=os.path.join(save_dir,'input',os.path.basename(raw_img_path).split('.')[0]+'_input.png')
    raw_img.save(input_path)


def read_xml(xml_path):
    tree=ET.parse(xml_path)
    root=tree.getroot()

    xmin = ymin = xmax = ymax = 0
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
    
    return xmin, ymin, xmax, ymax


def buid_batch_mask_png(root_dir,save_dir):
    img_dir=os.path.join(root_dir,'JPEGData')
    label_dir=os.path.join(root_dir,'label')

    cases=os.listdir(img_dir)
    for case in tqdm(cases,desc="Processing cases"):
        if os.path.isdir(os.path.join(img_dir,case)):
            imgs=os.listdir(os.path.join(img_dir,case))
            for img in imgs:
                img_path=os.path.join(img_dir,case,img)
                xml_name=img.split('.')[0]+'.xml'
                xml_path=os.path.join(label_dir,case,xml_name)
                (xmin, ymin, xmax, ymax)=read_xml(xml_path)
                out_mask_input(img_path,(xmin, ymin, xmax, ymax),save_dir)


buid_batch_mask_png('/home/wy/dataset/topicData/firstData/','/home/wy/dataset/topicData/make_dataset_v015')


    

    