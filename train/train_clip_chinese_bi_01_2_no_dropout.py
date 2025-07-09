import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from tqdm import tqdm
from torchvision.transforms import transforms


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算 pt
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal loss 计算
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CustomImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        初始化数据集
        :param file_path: JSONL 文件路径，每行是一个 JSON 对象
        :param transform: 可选的预处理或数据增强函数
        """
        self.classes_dict = {'Q-1': 0, 'Q0': 0, 'Q1':0, 'Q2':1}
        self.classes = [0, 1]
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取一条 JSON 数据
        item = self.data[idx]
        
        # 提取 label 与 ctype
        label = item.get("label", None)
        ctype = item.get("ctype", None)
        
        # 从 content 中提取关键信息
        content_text = None
        title = None
        tags = None
        if "content" in item and "meta" in item["content"]:
            meta_val = item["content"]["meta"].get("value", {})
            content_text = meta_val.get("content", None)
            title = meta_val.get("title", None)
            tags = meta_val.get("tag", None)
            content = item["content"]
        
        # 提取 cover 文件路径
        cover_path = None

        if "cover" in content and isinstance(content["cover"], dict):
            cover_path = content["cover"].get("value", None)
            cover_path = cover_path.replace("/tmp/dataset/", "/mnt/data/leaderboard/56da1986c432e0fff72fb039491c3548/")
            if not cover_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
                return self.__getitem__((idx + 1) % len(self.data))  # 递归调用下一个图片

        # 提取 video 文件路径
        video_path = None
        if "video" in content and isinstance(content["video"], dict):
            video_path = content["video"].get("value", None)
        
        # 构造返回的字典
        result = {
            "id": item.get("id", None),
            "label": self.classes_dict[label],
            "ctype": ctype,
            "content": content_text,
            "title": title,
            "tag": tags,
            "cover": cover_path,
            "video": video_path
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result

# 定义微调模型：冻结 CLIP 模型参数，仅训练分类器层
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        """
        :param clip_model: 预训练的 CLIP 模型
        :param num_classes: 类别数量
        """
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.in_features = 1024

        # 冻结 CLIP 模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 假设图像编码器输出为 512 维，添加新的分类层
        # self.classifier = nn.Linear(self.in_features, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, inputs):
        # 使用 CLIP 图像编码器获得图像 embedding
        # with torch.no_grad():
            # image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        outputs = self.clip_model(**inputs)
        text_embedding = outputs.text_embeds
        image_embedding = outputs.image_embeds
        fused_vector_concat = torch.cat((text_embedding, image_embedding), dim=1)

        # image_features: [batch_size, 512]
        logits = self.classifier(fused_vector_concat)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 CLIP 模型和处理器
    model_name = "chinese-clip-vit-base-patch16"

    clip_model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)

    # checkpoint = "./models/clip_finetuned_classifier_epoch1.pth"
    checkpoint = None

    # 定义图像预处理函数：利用 CLIPProcessor 的 feature_extractor
    def clip_transform(image):
        # 处理单张图像，返回 pixel_values（张量，形状 [3, 224, 224]）
        inputs = processor(images=image, return_tensors="pt", padding=True)
        # inputs["pixel_values"] 的 shape 为 [1, 3, 224, 224]，去除 batch 维度
        return inputs["pixel_values"].squeeze(0)
    
    # 构造数据集，假设数据目录为 "dataset"
    dataset = CustomImageDataset(file_path='datasets/raw/index_train_auxiliary.jsonl', transform=None)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    
    num_classes = len(dataset.classes)
    model = CLIPClassifier(clip_model, num_classes)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    model = model.to(device)

    # 定义损失和优化器，只优化分类器参数
    num_epochs = 2
    criterion = nn.CrossEntropyLoss()

        # 使用 Focal Loss 代替 CrossEntropyLoss
    criterion = FocalLoss(alpha=1, gamma=2)

    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    ckpt_path = "models_chinese_bi_01_2_0409_no_dropout"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        print(f"目录 '{ckpt_path}' 不存在，现已创建。")
    else:
        print(f"目录 '{ckpt_path}' 已存在。")
        
    for epoch in range(num_epochs):
        print(print(optimizer.param_groups[0]["lr"]))
        model.train()
        running_loss = 0.0
        for item in tqdm(dataloader):

            labels = item["label"]
            texts = item["title"]
            image_path = item["cover"]

            images = [Image.open(i) for i in image_path]

            inputs = processor(text=texts, 
                images=images, 
                return_tensors="pt", 
                padding=True).to(device)

            labels = torch.tensor(labels)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)
            print("loss:", loss.item())

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), f"{ckpt_path}/clip_finetuned_classifier_epoch{epoch+1}.pth")

    
    # 保存微调后的模型参数
    torch.save(model.state_dict(), f"{ckpt_path}/clip_finetuned_classifier_last.pth")
    print("模型已保存！")
    
if __name__ == "__main__":
    main()
