import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch.nn import DataParallel
import os
import matplotlib.pyplot as plt

from Data import our_dataloader
from Model.struct import multiboxloss
from Model.Retina import retina
from Config import _C

__all__ = ['train']

class Trainer_retina(object):

    def __init__(self):
        self.iterations = _C.MAX_ITER
        self.batch_size = _C.BATCH_SIZE1
        self.train_devices = [0]
        self.model_save_root = _C.RETINA_SAVE_DIR

        if not os.path.exists(self.model_save_root):
            os.mkdir(self.model_save_root)
        self.model_save_step = _C.RETINA_SAVE_STEP

        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.scheduler = None

    def __call__(self):
        """
        训练器使用, 传入 模型 与数据集.
        :param model:
        :param dataset:
        :return:
        """
        model=retina()
        if not isinstance(model, nn.DataParallel):
            # raise TypeError('请用 DataParallel 包装模型. eg: model = DataParallel(model, device_ids=[0,1,2]),使用device_ids指定需要使用的gpu')
            model = DataParallel(model, device_ids=self.train_devices)
        self.model = model
        data_loader = our_dataloader()
        
        print(' Max_iter = {}, Batch_size = {}'.format(self.iterations, self.batch_size))
        print(' Model will train on cuda:{}'.format(self.train_devices))

        num_gpu_use = len(self.train_devices)
        if (self.batch_size % num_gpu_use) != 0:
            raise ValueError(
                'You use {} gpu to train , but set batch_size={}'.format(num_gpu_use, data_loader.batch_size))

        self.set_lossfunc()
        self.set_optimizer()
        self.set_scheduler()

        print("Set optimizer : {}".format(self.optimizer))
        print("Set scheduler : {}".format(self.scheduler))
        print("Set lossfunc : {}".format(self.loss_func))


        print(' Start Train......')
        print(' -------' * 20)
        losses=[]

        for iteration, (images, boxes, labels, image_names, whs) in enumerate(data_loader):
            iteration+=1
            boxes, labels = boxes.to(_C.DEVICE), labels.to(_C.DEVICE)
            cls_logits, bbox_preds = self.model(images,image_names, whs)
            reg_loss, cls_loss = self.loss_func(cls_logits, bbox_preds, labels, boxes)

            reg_loss = reg_loss.mean()
            cls_loss = cls_loss.mean()
            loss = reg_loss + cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            if iteration % 10 == 0:
                print('Iter : {}/{} | Lr : {} | Loss : {:.4f} | cls_loss : {:.4f} | reg_loss : {:.4f}'.format(iteration, self.iterations, lr, loss.item(), cls_loss.item(), reg_loss.item()))

            if iteration % self.model_save_step == 0:
                losses.append(loss)
                torch.save(model.module.state_dict(), '{}/model_{}.pkl'.format(self.model_save_root, iteration))
            
            if iteration > self.iterations:
                break
        plt.plot(losses)
        plt.title('train loss')
        plt.xlabel('iter(100)')
        plt.ylabel('loss')
        plt.savefig('train_loss.png')
        plt.show()
        
        return True

    def set_optimizer(self):
        """
        配置优化器
        :param lr:              初始学习率,  默认0.001
        :param momentum:        动量, 默认 0.9
        :param weight_decay:    权重衰减,L2, 默认 5e-4
        :return:
        """

        lr= _C.LEARNING_RATE1
        momentum = _C.MOMENTUM1
        weight_decay = _C.WEIGHT_DECAY1

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)

    def set_lossfunc(self):
        """
        配置损失函数
        :param neg_pos_ratio:   负正例 比例,默认3, 负例数量是正例的三倍
        :return:
        """
        neg_pos_ratio = _C.NEG_POS_RATIO
        self.loss_func = multiboxloss()
        # print(' Trainer set loss_func : {}, neg_pos_ratio = {}'.format('multiboxloss', neg_pos_ratio))

    def set_scheduler(self):
        """
        配置学习率衰减策略
        :param lr_steps:    默认 [80000, 10000],当训练到这些轮次时,学习率*gamma
        :param gamma:       默认 0.1,学习率下降10倍
        :return:
        """
        lr_steps = _C.SCHEDULER_LR_STEPS
        gamma = _C.SCHEDULER_GAMMA
        self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                     milestones=lr_steps,
                                     gamma=gamma)
        # print(' Trainer set scheduler : {}, lr_steps={}, gamma={}'.format('MultiStepLR', lr_steps, gamma))

def train():
    trainer=Trainer_retina()
    trainer()