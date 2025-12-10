import os
import torch.utils.data
from PIL import Image
from torchvision import transforms

from Config import _C


__all__ = ['vocdataset']

class vocdataset(torch.utils.data.Dataset):

    def __init__(self, is_train=True, keep_difficult=False):
        """VOC格式数据集
        Args:
            data_dir: VOC格式数据集根目录，该目录下包含：
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
            split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
        """
        # 类别
        self.data_dir = _C.DATASET_DIR
        self.is_train = is_train
        self.split = 'train'       # train     对应于ImageSets/Detection/train.txt
        if not self.is_train:
            self.split = 'val'    # test      对应于ImageSets/Detection/test.txt
        image_sets_file = os.path.join(self.data_dir, "ImageSets", 'Main', "{}.txt".format(self.split))
        # 从train.txt 文件中读取图片 id 返回ids列表
        self.ids = self._read_image_ids(image_sets_file)

    def __getitem__(self, index):
        image_name = self.ids[index]
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        labels = self._get_annotation(image_name)
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)

        return image, labels

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                for i in range(0,9):
                    ids.append(line.rstrip()+f'_{i}')
        return ids
    

    # 解析xml，返回 boxes， labels， is_difficult   numpy.array格式
    def _get_annotation(self, image_name):
        n=int(image_name.split('e')[1].split('_')[0])
        if n>=0 and n<=138:
            label=0
        elif n>=139 and n<=171:
            label=1
        elif n>=172 and n<=186:
            label=2
        else:
            raise RuntimeError('不存在的label！')
        
        label=torch.tensor(label)
        #label_onehot = F.one_hot(label, _C.DATA.DATASET.NUM_CLASSES)
        return label

    # 读取图片数据，返回Image.Image
    def _read_image(self, image_id):
        case_name=image_id.split('_')[0]
        slice_id=image_id.split('_')[1]
        image_file = os.path.join(self.data_dir, "MultiAngle", case_name,f'{case_name}_{slice_id}',"{}.jpg".format(image_id))
        image = self.transform(Image.open(image_file).convert("RGB"))
        return image
    
    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((_C.IMAGE_SIZE,_C.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,],
                std=[0.229,]
            )
        ])(img)


if __name__=='__main__':
    '''train_set=rgbdataset()
    val_set=rgbdataset(is_train=False)
    print(len(train_set))
    print(len(val_set))
    x,y=train_set[0]
    print(x.shape)
    print(y.shape)'''
    pass