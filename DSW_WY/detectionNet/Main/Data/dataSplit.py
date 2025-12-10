import os
import numpy as np
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET

from Config import _C

    
def detect_label(xml_path,direction,num):
    tree=ET.parse(xml_path)
    root = tree.getroot()
    direction_elem = root.find(direction)
    begin=int(direction_elem.find('begin').text)
    end=int(direction_elem.find('end').text)
    #print(begin,end)
    if int(num)>=begin and int(num)<=end:
        return 'Positive'
    else:
        return 'Negative'


def split(dataset_dir=_C.DATASET_DIR, test_ratio=_C.TEST_RATIO): 
    
    csv_path=os.path.join(dataset_dir,'dataset_csv')
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
        print('creating dataset_csv...')

    # 划分测试集和训练集
    train_set = []
    test_set = []
    classData_dir=os.path.join(dataset_dir,'classData')
    classes = os.listdir(classData_dir)
    for class_index, classname in enumerate(classes):
        # 读取所有视频路径
        cases = os.listdir(os.path.join(classData_dir, classname))
        # 打乱视频名称
        np.random.shuffle(cases)
        # 确定测试集划分点
        split_size = int(len(cases) * test_ratio)
        
        for i in tqdm(range(len(cases)),desc=f'Processing class {classname}'):
            case_name=cases[i].split('.')[0]  #case0
            case_type='test' if i<split_size else 'train'
            jpg_direction=['axial','sagittal','coronal']
            for direction in jpg_direction:
                jpgs=os.listdir(os.path.join(dataset_dir,'JPEGData',case_name,direction))
                for j in range(0,len(jpgs)):
                    #img_name=f'{case_name}_{direction}_{j}.jpg'
                    img_name=jpgs[j]
                    img_path=os.path.join(dataset_dir,'JPEGData',case_name,direction,img_name)
                    xml_path=os.path.join(dataset_dir,'label',f'{case_name}.xml')
                    jpg_num=jpgs[j].split('.')[0].split('_')[2]
                    img_label=detect_label(xml_path,direction,jpg_num)  
                    info=[f'{case_name}_{direction}',img_label,img_path]
                    if case_type=='test':
                        test_set.append(info)
                    else:
                        train_set.append(info) 

    with open(csv_path + '/' + 'train.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in train_set]))
    with open(csv_path + '/' + 'test.csv', 'w') as f:
        f.write('\n'.join([','.join(line) for line in test_set]))


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 make_train_test.py -i path/to/dataset -o path/to/current_project_dir -s 0.2')
    parser.add_argument('-i', '--dataset_dir', help='path to dataset', default=_C.DATASET_DIR)
    parser.add_argument('-s', '--test_ratio', help='ratio of test sets', default=_C.TEST_RATIO)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    split(**vars(args))