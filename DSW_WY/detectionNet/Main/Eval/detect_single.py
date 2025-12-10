import os
from PIL import Image
import numpy as np
import torch
from vizer.draw import draw_boxes
import matplotlib.pyplot as plt

from Model.Retina import retina
from Config import _C
from Transforms import transform


__all__=['detect']

class DetectorSingle(object):
    def __init__(self):
        self.model=retina()
        
    @torch.no_grad()
    def __call__(self,img_name):
        self.model.to(_C.DEVICE)
        self.model.load_pretrained_weight(os.path.join(_C.RETINA_SAVE_DIR,_C.DETECT_RETINA))
        self.model.eval()
        img_path=os.path.join(_C.DATASET_DIR1,'JPEGImages',img_name)
        image = Image.open(img_path).convert("RGB")
        w, h = image.width, image.height
        wh=torch.tensor([w,h])
        wh=wh.unsqueeze(dim=0)
        images_tensor = transform(is_train=False)(np.array(image))[0].unsqueeze(0)
        images_tensor=images_tensor.to(_C.DEVICE)
        detections=self.model.forward_with_postprocess(images_tensor,(img_name.split('.')[0],),wh)[0]
        boxes, labels, scores = detections
        boxes, labels, scores = boxes.to('cpu').numpy(), labels.to('cpu').numpy(), scores.to('cpu').numpy()

        boxes[:, 0::2] *= (w / _C.IMAGE_SIZE1)
        boxes[:, 1::2] *= (h / _C.IMAGE_SIZE1)

        indices = scores > _C.TEST_CONFIDENCE_THRESHOLD
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        drawn_image = draw_boxes(image=image, boxes=boxes, labels=labels,
                                 scores=scores, class_name_map=_C.CLASS_NAMES1).astype(np.uint8)
        
        plt.imsave('detect_result.jpg',drawn_image)
        plt.imshow(drawn_image)
        plt.show()
        return drawn_image


def detect(img_name):
    detector=DetectorSingle()
    drawn_image=detector(img_name)
    return drawn_image