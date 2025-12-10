import numpy as np
import torch

from Utils import corner_form_to_center_form, center_form_to_corner_form
from Config import _C

class priorbox:
    """
    Retainnet anchors
    """
    def __init__(self):
        self.features_maps = _C.ANCHORS_FEATURE_MAPS
        self.anchor_sizes = _C.ANCHORS_SIZES
        self.ratios = np.array(_C.ANCHORS_RATIOS)
        self.scales = np.array(_C.ANCHORS_SCALES)
        self.image_size = _C.IMAGE_SIZE1
        self.clip = _C.ANCHORS_CLIP

    def __call__(self):
        priors = []
        for k , (feature_map_w, feature_map_h) in enumerate(self.features_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h

                    size = self.anchor_sizes[k]/self.image_size    

                    sides_square = self.scales * size   
                    for side_square in sides_square:
                        priors.append([cx, cy, side_square, side_square])   

                    sides_long = sides_square*2**(1/2)  
                    for side_long in sides_long:
                        priors.append([cx, cy, side_long, side_long/2]) 
                        priors.append([cx, cy, side_long/2, side_long])

        priors = torch.tensor(priors)
        if self.clip:   
            priors = center_form_to_corner_form(priors) 
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) 
        return priors

if __name__ == '__main__':
    anchors = priorbox()()
    print(anchors[-10:])
    print(len(anchors))
