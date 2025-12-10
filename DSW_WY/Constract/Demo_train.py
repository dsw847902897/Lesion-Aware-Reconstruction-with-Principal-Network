import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

# 1. 自定义数据集类
class MedicalLesionDataset(Dataset):
    def __init__(self, root_dir, transform=None, pos_pair_ratio=0.5):
        """
        自定义医学病变数据集
        
        参数:
            root_dir: 数据集根目录
            transform: 数据增强变换
            pos_pair_ratio: 正样本对的比例
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pos_pair_ratio = pos_pair_ratio
        
        self.positive_samples = [os.path.join(root_dir, 'positive', f) 
                               for f in os.listdir(os.path.join(root_dir, 'positive'))]
        self.negative_samples = [os.path.join(root_dir, 'negative', f) 
                               for f in os.listdir(os.path.join(root_dir, 'negative'))]
        
        self.all_samples = self.positive_samples + self.negative_samples
        self.labels = [1]*len(self.positive_samples) + [0]*len(self.negative_samples)
        
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        img_path = self.all_samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label
    
    def get_pair(self, idx):
        """
        获取正负样本对
        返回:
            anchor: 锚点图像
            positive: 正样本(相同类别)
            negative: 负样本(不同类别)
        """
        anchor_img, anchor_label = self.__getitem__(idx)
        
        # 选择正样本
        if np.random.rand() < self.pos_pair_ratio and sum(self.labels) > 0:
            # 正样本对
            pos_indices = [i for i, lbl in enumerate(self.labels) 
                         if lbl == anchor_label and i != idx]
            if len(pos_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                positive_img, _ = self.__getitem__(pos_idx)
                pair_label = 1  # 正样本对标签
            else:
                # 如果没有相同类别的样本，使用负样本
                pos_indices = [i for i, lbl in enumerate(self.labels) 
                              if lbl != anchor_label]
                pos_idx = np.random.choice(pos_indices)
                positive_img, _ = self.__getitem__(pos_idx)
                pair_label = 0  # 负样本对标签
        else:
            # 负样本对
            pos_indices = [i for i, lbl in enumerate(self.labels) 
                         if lbl != anchor_label]
            pos_idx = np.random.choice(pos_indices)
            positive_img, _ = self.__getitem__(pos_idx)
            pair_label = 0  # 负样本对标签
            
        return anchor_img, positive_img, pair_label

# 2. 数据增强
def get_transforms(input_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 3. 对比学习评分网络
class ResNetContrastiveScorer(nn.Module):
    def __init__(self, feature_dim=128, temperature=0.1, pretrained=True):
        super(ResNetContrastiveScorer, self).__init__()
        self.temperature = temperature
        
        # 使用预训练的ResNet50作为编码器
        self.encoder = models.resnet50(pretrained=pretrained)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # 移除最后的全连接层
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, feature_dim)
        )
        
        # 评分头
        self.scorer = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取特征
        features = self.encoder(x)
        
        # 投影到对比学习空间
        projected = self.projector(features)
        
        # 计算评分
        score = self.scorer(features)
        
        return projected, score
    
    def contrastive_loss(self, proj1, proj2, labels):
        """
        计算对比损失
        
        参数:
            proj1: 投影特征1
            proj2: 投影特征2
            labels: 样本对标签 (1表示正样本对，0表示负样本对)
        """
        # 计算余弦相似度
        sim = F.cosine_similarity(proj1, proj2, dim=-1) / self.temperature
        
        # 正样本对的损失
        pos_loss = -torch.mean(sim[labels == 1])
        
        # 负样本对的损失
        neg_loss = torch.mean(torch.exp(sim[labels == 0]))
        
        return pos_loss + neg_loss

# 4. 训练函数
def train_model(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_score = 0.0
    
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        proj1, score1 = model(img1)
        proj2, score2 = model(img2)
        
        # 计算损失
        loss = model.contrastive_loss(proj1, proj2, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        running_score += (score1.mean() + score2.mean()).item() / 2
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(img1)}/{len(dataloader.dataset)}]'
                  f'\tLoss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(dataloader)
    avg_score = running_score / len(dataloader)
    return avg_loss, avg_score

# 5. 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            proj1, score1 = model(img1)
            proj2, score2 = model(img2)
            
            loss = model.contrastive_loss(proj1, proj2, labels)
            
            running_loss += loss.item()
            running_score += (score1.mean() + score2.mean()).item() / 2
    
    avg_loss = running_loss / len(dataloader)
    avg_score = running_score / len(dataloader)
    return avg_loss, avg_score

# 6. 主函数
def main():
    # 参数设置
    data_dir = '/home/wy/dataset/topicData/mask_dataset/cnnDataset/train'  # 替换为您的数据集路径
    batch_size = 32
    num_epochs = 50
    feature_dim = 128
    temperature = 0.1
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据
    train_transform, val_transform = get_transforms()
    
    train_dataset = MedicalLesionDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = MedicalLesionDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 初始化模型
    model = ResNetContrastiveScorer(
        feature_dim=feature_dim,
        temperature=temperature
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_score = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_score = train_model(model, train_loader, optimizer, device, epoch)
        val_loss, val_score = evaluate_model(model, val_loader, device)
        
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Score: {train_score:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}')
        
        # 保存最佳模型
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
    
    print('Training completed!')

if __name__ == '__main__':
    main()