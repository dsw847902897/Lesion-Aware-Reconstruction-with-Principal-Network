import torch.nn as nn
import math
import torch
from torch.nn import Sequential, Linear, ReLU, Softmax, Module

from Config import _C

__all__=['PolicyNet','ValueNet','encodercnn']

class PolicyNet(Module):
    def __init__(self, state_dim=_C.STATE_DIM, hidden_dim=_C.HIDDEN_DIM, action_card=_C.ACTION_CARD):
        super(PolicyNet, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                #Linear(hidden_dim, hidden_dim),
                                #ReLU(),
                                Linear(hidden_dim, action_card),
                                Softmax(dim=0))

    def forward(self, x):
        return self.model(x)


class ValueNet(Module):
    def __init__(self, state_dim=_C.STATE_DIM, hidden_dim=_C.HIDDEN_DIM):
        super(ValueNet, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, 1))

    def forward(self, x):
        return self.model(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EncoderCNN(nn.Module):
    def __init__(self, n_class=_C.NUM_CLASSES, input_size=_C.IMAGE_SIZE, width_mult=1.):
        super(EncoderCNN, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def encodercnn():
    model=EncoderCNN().to(_C.DEVICE)
    model.load_state_dict(torch.load(_C.PRETRAINED_MODEL_PATH))
    return model
    
if __name__ == '__main__':
    '''net = GlobalCNN()
    train_set=rgbdataset()
    val_set=rgbdataset(is_train=False)
    params = {'batch_size':_C.POINT_CHOOSE_BS, 'shuffle': _C.DATASET_SHUFFLE, 'num_workers': _C.DATASET_NUM_WORKERS, 'pin_memory': _C.DATASET_PIN_MEMORY}
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(val_set, **params)
    for x,y in train_loader:
        print(x.shape)             #torch.Size([32, 3, 512, 512])
        print(y.shape)             #torch.Size([32])
        output=net(x)
        print(output)        #torch.Size([32, 3])
        yy=output.max(1, keepdim=True)[1].squeeze()
        print(yy)'''
    pass