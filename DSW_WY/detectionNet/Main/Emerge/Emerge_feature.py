import os
from torchvision import transforms
from PIL import Image
import torch
import nibabel as nib
import numpy as np
import cv2

from Config import _C
from Model.Mobile import mobile
from Model.struct import postprocessor
from Policy.env import SelectionEnv
from Policy.ActorCritic import ActorCritic

__all__=['emerge_feature']


def emerge_feature(cls_logits,bbox_pred,img_names,img_whs,emerge_ratio):
    emerger=Emeger()
    emerged_feature=emerger(cls_logits,bbox_pred,img_names,img_whs,emerge_ratio)
    return emerged_feature

class Emeger(object):
    def __init__(self):
        self.train_stage=_C.TRAIN_STAGE
        self.device=_C.DEVICE
        self.model=mobile().to(self.device)
        self.pretrained_weight_path=os.path.join(_C.MOBILE_TRAINED_DIR,_C.MOBILE_PICK)
        self.model.load_state_dict(torch.load(self.pretrained_weight_path))
        self.model.eval()
        self.postprocessor = postprocessor()

        self.env = SelectionEnv()
        self.agent = ActorCritic()
    

    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((_C.IMAGE_SIZE2,_C.IMAGE_SIZE2)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,],
                std=[0.229,]
            )
        ])(img)

    def make_feature(self,output,cls_logits,ratio):
        b,f,n=cls_logits.shape
        feature=torch.zeros((b,f,n))
        for i, value in enumerate(output):
            two_dim_tensor = torch.zeros((f, n))
            two_dim_tensor[:, int(value) + 1] = 1
            two_dim_tensor[two_dim_tensor == 0] = ratio
            feature[i, :, :] = two_dim_tensor
        return feature
    

    def get_rgbImg(self,img_name,x,y,z):
        case=img_name.split('_')[0]
        nii_path=os.path.join(_C.DATASET_DIR1, 'Data', case,f'{case}.nii')
        nifti_img = nib.load(nii_path)
        img_data = nifti_img.get_fdata()
        r_img=img_data[:,:, z]
        if np.max(r_img) == np.min(r_img):
            r_img = (r_img - np.min(r_img))* 255.0
        else:
            r_img = (r_img - np.min(r_img)) / (np.max(r_img) - np.min(r_img)) * 255.0
        r_img=r_img.astype(np.uint8)
        height, width = r_img.shape

        g_img=img_data[:,y,:]
        if np.max(g_img) == np.min(g_img):
            g_img = (g_img - np.min(g_img))* 255.0
        else:
            g_img = (g_img - np.min(g_img)) / (np.max(g_img) - np.min(g_img)) * 255.0
        g_img = g_img.astype(np.uint8)
        g_img = cv2.resize(g_img, (width, height), interpolation=cv2.INTER_LINEAR)

        b_img=img_data[x,:,:]
        if np.max(b_img) == np.min(b_img):
            b_img = (b_img - np.min(b_img))* 255.0
        else:
            b_img = (b_img - np.min(b_img)) / (np.max(b_img) - np.min(b_img)) * 255.0
        b_img = b_img.astype(np.uint8)
        b_img = cv2.resize(b_img, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb_image = np.stack([r_img, g_img, b_img], axis=-1)
        rgb_image = self.transform(Image.fromarray(np.uint8(rgb_image))).unsqueeze(0)
        return rgb_image


    def __call__(self,cls_logits,bbox_pred,img_names,img_whs,emerge_ratio):
        if self.train_stage == 1:
            emerged_feature=self.train_stage1(cls_logits,img_names,emerge_ratio)
        if self.train_stage == 2:
            emerged_feature=self.train_stage2(cls_logits,bbox_pred,img_names,img_whs,emerge_ratio)
        if self.train_stage == 3:
            emerged_feature=self.train_stage3(cls_logits,bbox_pred,img_names,img_whs,emerge_ratio)

        return emerged_feature


    def train_stage1(self,cls_logits,img_names,emerge_ratio):
        rgb_imgs=[]
        for img_name in img_names:
            rgb_path=os.path.join(_C.DATASET_DIR1,'MultiAngle',img_name.split('_')[0],img_name,f'{img_name}_4.jpg')
            rgb_img = self.transform(Image.open(rgb_path).convert("RGB"))
            rgb_imgs.append(rgb_img)
        rgb_imgs=torch.stack(rgb_imgs,0).to(self.device)
        labels=self.model(rgb_imgs).max(1, keepdim=True)[1].squeeze(dim=0)
        feature_rgb=self.make_feature(labels,cls_logits,emerge_ratio).to(self.device)
        emerged_feature=torch.mul(cls_logits,feature_rgb)
        return emerged_feature


    def train_stage2(self,cls_logits,bbox_pred,img_names,img_whs,emerge_ratio):
        cls_logits1=self.train_stage1(cls_logits,img_names,emerge_ratio)
        b,f,n=cls_logits1.shape
        features_rgb=[]
        results=self.postprocessor(cls_logits1,bbox_pred)
        #print(results)
        for img_name,result,img_wh in zip(img_names,results,img_whs):
            box,label,score=result
            if len(box) != 0:
                box[:, 0::2] *= (img_wh[0] / _C.IMAGE_SIZE1)
                box[:, 1::2] *= (img_wh[1] / _C.IMAGE_SIZE1)
                x=int((box[0][0]+box[0][2])/2)
                y=int((box[0][1]+box[0][3])/2)
                z=int(img_name.split('_')[1])
                rgb_img=self.get_rgbImg(img_name,x,y,z).to(self.device)
                y_pred=self.model(rgb_img).max(1, keepdim=True)[1].item()
                feature_rgb = torch.full((f, n), emerge_ratio)
                feature_rgb[:,y_pred+1]=1
                features_rgb.append(feature_rgb)
            else:
                rgb_path=os.path.join(_C.DATASET_DIR1,'MultiAngle',img_name.split('_')[0],img_name,f'{img_name}_4.jpg')
                rgb_img = self.transform(Image.open(rgb_path).convert("RGB")).unsqueeze(0).to(self.device)
                y_pred=self.model(rgb_img).max(1, keepdim=True)[1].item()
                feature_rgb = torch.full((f, n), emerge_ratio)
                feature_rgb[:,y_pred+1]=1
                features_rgb.append(feature_rgb)
        features_rgb = torch.stack(features_rgb, dim=0).to(self.device)
        emerged_feature=torch.mul(cls_logits,features_rgb)
        return emerged_feature

    def train_stage3(self,cls_logits,bbox_pred,img_names,img_whs,emerge_ratio):
        cls_logits1=self.train_stage1(cls_logits,img_names,emerge_ratio)
        b,f,n=cls_logits1.shape
        results=self.postprocessor(cls_logits1,bbox_pred)
        features_rgb=[]
        for img_name,result,img_wh in zip(img_names,results,img_whs):
            box,label,score=result
            if len(box) != 0:
                box[:, 0::2] *= (img_wh[0] / _C.IMAGE_SIZE1)
                box[:, 1::2] *= (img_wh[1] / _C.IMAGE_SIZE1)
                xmin=box[0][0]
                xmax=box[0][2]
                ymin=box[0][1]
                ymax=box[0][3]
                z=int(img_name.split('_')[1])
                xys=[]
                for i in range(0,3):
                    for j in range(0,3):
                        x=int(xmin+(xmax-xmin)*(2*j+1)/6)
                        y=int(ymin+(ymax-ymin)*(2*i+1)/6)
                        xys.append([x,y])
                img_batch=[]
                for i in range (0,len(xys)):
                    img_batch.append(self.get_rgbImg(img_name,xys[i][0],xys[i][1],z))
                tensor_batch=torch.stack(img_batch,axis=0).squeeze(dim=1)
                state = self.env.reset(tensor_batch.detach())
                action = self.agent.take_action(state)     #0~8
                rgb_img=img_batch[action].to(self.device)
                y_pred=self.model(rgb_img).max(1, keepdim=True)[1].item()
                feature_rgb = torch.full((f, n), emerge_ratio)
                feature_rgb[:,y_pred+1]=1
                features_rgb.append(feature_rgb)
            if len(box) ==0:
                rgb_path=os.path.join(_C.DATASET_DIR1,'MultiAngle',img_name.split('_')[0],img_name,f'{img_name}_4.jpg')
                rgb_img = self.transform(Image.open(rgb_path).convert("RGB")).unsqueeze(0).to(self.device)
                y_pred=self.model(rgb_img).max(1, keepdim=True)[1].item()
                feature_rgb = torch.full((f, n), emerge_ratio)
                feature_rgb[:,y_pred+1]=1
                features_rgb.append(feature_rgb)
        features_rgb = torch.stack(features_rgb, dim=0).to(self.device)
        emerged_feature=torch.mul(cls_logits,features_rgb)
        return emerged_feature




