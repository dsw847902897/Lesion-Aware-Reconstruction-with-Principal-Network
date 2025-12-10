from Utils.Boxs_op import center_form_to_corner_form, assign_priors,\
    corner_form_to_center_form, convert_boxes_to_locations
from .Transforms_utils import *
from Config import _C

__all__ = ['transform', 'targettransform']

class transform:
    """
    transfroms
    eg:
        transform = Tramsfrom(cfg,is_train=True)
    """
    def __init__(self, is_train):
        if is_train:
            self.transforms = [
                ConvertFromInts(),  
                PhotometricDistort(),   
                SubtractMeans(_C.DATA_PIXEL_MEAN),  
                DivideStds(_C.DATA_PIXEL_STD),    
                Expand(), 
                RandomSampleCrop(), 
                RandomMirror(),     
                ToPercentCoords(),  
                Resize(_C.IMAGE_SIZE1),

                ToTensor(),
            ]
        else:
            self.transforms = [
                Resize(_C.IMAGE_SIZE1),
                SubtractMeans(_C.DATA_PIXEL_MEAN),  
                DivideStds(_C.DATA_PIXEL_STD),  
                ToTensor()
            ]

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class targettransform:
    """
    targets_transfroms
    eg:
        transform = TargetTransform(cfg)
    """

    def __init__(self):
        from Model.struct import priorbox  

        self.center_form_priors = priorbox()()
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.center_variance = _C.ANCHORS_CENTER_VARIANCE
        self.size_variance = _C.ANCHORS_SIZE_VARIANCE
        self.iou_threshold = _C.ANCHORS_THRESHOLD 

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = assign_priors(gt_boxes, gt_labels,
                                      self.corner_form_priors,
                                      self.iou_threshold)
        boxes = corner_form_to_center_form(boxes)
        locations = convert_boxes_to_locations(boxes,
                                               self.center_form_priors,
                                               self.center_variance,
                                               self.size_variance)
        return locations, labels
