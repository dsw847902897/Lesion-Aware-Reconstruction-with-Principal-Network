from Data import vocdataset
from Data import our_dataloader, our_dataloader_val
import torch


def test_vocDataset():
    train_dataset=vocdataset()
    val_dataset=vocdataset(is_train=False)
    print(len(train_dataset))
    print(len(val_dataset))
    img,box,label,name,whs=train_dataset[0]
    print(img.shape)       #torch.Size([3, 580, 580])
    print(box.shape)       #torch.Size([64656, 4])
    print(label.shape)     #torch.Size([64656])
    print(name)
    print(whs.shape)       #torch.Size([2])
    print(whs)

def test_vocdataloader():
    train_loader=our_dataloader()
    val_loader=our_dataloader_val()
    for imgs,boxes,labels,names,whs in val_loader:
        print(imgs.shape)       #torch.Size([32, 3, 580, 580])     (B,C,W,H)
        print(boxes.shape)      #torch.Size([32, 64656, 4])
        print(labels.shape)     #torch.Size([32, 64656])
        print(names)
        print(whs)        #torch.Size([96, 2])

if __name__=='__main__':
    #test_vocDataset()
    test_vocdataloader()