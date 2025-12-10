from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import Sampler
from torch.utils.data.dataloader import default_collate   # 这个不用管，只是显示问题，实际可以使用

from .Dataset_VOC import vocdataset
#from .Dataset3D import dataset3d
from Config import _C

__all__ = ['our_dataloader', 'our_dataloader_val','dataloader_rgb','dataloader_rgb_val']

class BatchSampler_Our(Sampler):
    """
    重新定义了 批采样类 ，实现按指定迭代数进行批次提取，
    在取完一批次后没达到指定迭代数会进行循环，直到输出指定的批次数量。
    """

    def __init__(self, sampler, batch_size, max_iteration=100000000, drop_last=True):
        """
        数据加载,默认循环加载1亿次,几近无限迭代.
        每次迭代输出一个批次的数据.
        :param sampler:         采样器，传入 不同采样器 实现 不同的采样策略，    RandomSampler随机采样，SequentialSampler顺序采样
        :param batch_size:      批次大小
        :param max_iteration:   迭代次数
        :param drop_last:       是否弃掉最后的不够一批次的数据。True则弃掉；False保留，并返回，但是这一批次会小于指定批次大小。
        """
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(max_iteration, int) or isinstance(max_iteration, bool) or \
                max_iteration <= 0:
            raise ValueError("max_iter should be a positive integer value, "
                             "but got max_iter={}".format(max_iteration))

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.drop_last = drop_last

    def __iter__(self):
        iteration = 0

        while iteration <= self.max_iteration:
            batch = []
            for idx in self.sampler:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    iteration += 1
                    yield batch
                    batch = []

                    if iteration > self.max_iteration:
                        break

            if len(batch) > 0 and not self.drop_last:
                iteration += 1
                yield batch

                if iteration > self.max_iteration:
                    break

    def __len__(self):
        if self.drop_last:
            return self.max_iteration
        else:
            return self.max_iteration


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[3])
        whs=default_collate(transposed_batch[4])

        if self.is_train:
            boxes = default_collate(transposed_batch[1])
            labels = default_collate(transposed_batch[2])
        else:
            boxes = None
            labels = None
        return images, boxes, labels, img_ids, whs
    

def our_dataloader(batch_size=_C.BATCH_SIZE1,shuffle=_C.SHUFFLE1,num_workers=_C.NUM_WORKERS1,drop_last=False,max_iteration=100000000):

    dataset=vocdataset()

    if shuffle:
        sampler = RandomSampler(dataset)        # 随机采样器
    else:
        sampler = SequentialSampler(dataset)    # 顺序采样器
    batch_sampler = BatchSampler_Our(sampler=sampler,
                                     batch_size=batch_size,
                                     max_iteration=max_iteration,
                                     drop_last=drop_last)
    loader = DataLoader(dataset=dataset,batch_sampler=batch_sampler,num_workers=num_workers,collate_fn=BatchCollator(is_train=dataset.is_train))
    return loader

def our_dataloader_val(batch_size=_C.EVAL_BATCH_SIZE1,shuffle=_C.EVAL_SHUFFLE1,get_box_label=True,num_workers=_C.EVAL_NUM_WORKERS1,drop_last=False):
    dataset=vocdataset(is_train=False)
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
                        collate_fn=BatchCollator(is_train=get_box_label),drop_last=drop_last)
    return loader

'''def dataloader_rgb(batch_size=_C.TRAIN3.BATCH_SIZE, shuffle=_C.TRAIN3.SHUFFLE, num_workers=_C.TRAIN3.NUM_WORKERS,pin_memory=_C.TRAIN3.PIN_MEMORY):
    dataset=dataset3d()
    loader=DataLoader(dataset,batch_size,shuffle,num_workers=num_workers,pin_memory=pin_memory)
    return loader

def dataloader_rgb_val(batch_size=_C.EVAL3.BATCH_SIZE, shuffle=_C.EVAL3.SHUFFLE, num_workers=_C.EVAL3.NUM_WORKERS,pin_memory=_C.EVAL3.PIN_MEMORY):
    dataset=dataset3d(is_train=False)
    loader=DataLoader(dataset,batch_size,shuffle,num_workers=num_workers,pin_memory=pin_memory)
    return loader'''

