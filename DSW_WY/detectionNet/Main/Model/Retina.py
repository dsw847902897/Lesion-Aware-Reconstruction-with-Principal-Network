from torch import nn
import torch

from .base_models import build_resnet
from .struct import fpn, predictor, postprocessor
from Emerge import emerge_feature
from Config import _C

__all__=['retina']

class RetainNet(nn.Module):
    """
    :return cls_logits, torch.Size([B, 67995, num_classes])
            bbox_pred,  torch.Size([B, 67995, 4])
    """
    def __init__(self):
        super(RetainNet,self).__init__()
        self.resnet = _C.BASEMODEL
        self.num_classes = _C.NUM_CLASSES1
        self.num_anchors = _C.ANCHORS_NUMS
        expansion_list={
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        assert self.resnet in expansion_list

        self.backbone = build_resnet(self.resnet, pretrained=True)
        expansion = expansion_list[self.resnet]
        self.fpn = fpn(channels_of_fetures=[128*expansion, 256*expansion, 512*expansion])
        self.predictor = predictor(num_anchors=self.num_anchors, num_classes=self.num_classes)  # num_anchors 默认为9,与anchor生成相对应
        self.postprocessor = postprocessor()

        #self.w1 = nn.Parameter(torch.rand(1))
        self.w1 = _C.EMERGE_RATIO
        
    def load_pretrained_weight(self, weight_pkl):
        self.load_state_dict(torch.load(weight_pkl))
        
    def forward(self, x,img_names,img_whs):
        c3, c4, c5, p6, p7 = self.backbone(x)   # resnet输出五层特征图
        p3, p4, p5 = self.fpn([c3, c4, c5])     # 前三层特征图进FPN
        features = [p3, p4, p5, p6, p7]
        cls_logits, bbox_pred = self.predictor(features)    #cls_logits:torch.Size([32, 64656, 4])  (b,64656,num_classes)
        emerged_cls_logits=emerge_feature(cls_logits,bbox_pred,img_names,img_whs,self.w1)
        #print(cls_logits-emerged_cls_logits)
        #emerged_cls_logits=cls_logits
        #print(detections)
        return emerged_cls_logits, bbox_pred
        #return cls_logits, bbox_pred

    def forward_with_postprocess(self, images,image_names,whs):
        """
        前向传播并后处理
        :param images:
        :return:
        """
        cls_logits, bbox_pred = self.forward(images,image_names,whs)
        detections = self.postprocessor(cls_logits, bbox_pred)
        return detections


def retina():
    model = RetainNet()
    return model