import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from torchvision.transforms import transforms
import requests
from sklearn.metrics import accuracy_score, recall_score, classification_report
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        self.classes_dict =  {'Q-1': 0, 'Q0': 1, 'Q1':1, 'Q2':2}
        self.classes = [0, 1, 2]
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


class AttentionFusion(nn.Module):
    def __init__(self, input_dim=512):
        super(AttentionFusion, self).__init__()
        # 定义一个线性层来计算注意力权重
        self.attention_layer = nn.Linear(input_dim, 1, bias=False)

    def forward(self, v1, v2):
        # 输入向量 v1 和 v2 的形状为 (batch_size, 512)
        # 将两个向量堆叠起来，形状变为 (batch_size, 2, 512)
        stacked_vectors = torch.stack([v1, v2], dim=1)  # [batch_size, 2, 512]

        # 计算每个向量的注意力权重
        attention_weights = self.attention_layer(stacked_vectors)  # [batch_size, 2, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, 2]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, 2]

        # 根据注意力权重加权融合两个向量
        fused_vector = attention_weights[:, 0].unsqueeze(-1) * v1 + attention_weights[:, 1].unsqueeze(-1) * v2
        return fused_vector


class AttentionFusion3d(nn.Module):
    def __init__(self, input_dim=512):
        super(AttentionFusion3d, self).__init__()
        # 定义一个线性层来计算注意力权重
        self.attention_layer = nn.Linear(input_dim, 1, bias=False)

    def forward(self, v1, v2, v3):
        # 输入向量 v1, v2, v3 的形状为 (batch_size, 512)
        # 将三个向量堆叠起来，形状变为 (batch_size, 3, 512)
        stacked_vectors = torch.stack([v1, v2, v3], dim=1)  # [batch_size, 3, 512]

        # 计算每个向量的注意力权重
        attention_weights = self.attention_layer(stacked_vectors)  # [batch_size, 3, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, 3]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, 3]

        # 根据注意力权重加权融合三个向量
        fused_vector = (
            attention_weights[:, 0].unsqueeze(-1) * v1 +
            attention_weights[:, 1].unsqueeze(-1) * v2 +
            attention_weights[:, 2].unsqueeze(-1) * v3
        )
        return fused_vector
    

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


def evaluate(model, dataloader, processor, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for item in tqdm(dataloader, desc="Evaluating"):

            labels = item["label"]
            texts = item["title"]
            img_paths = item["cover"]
            tags = item["tag"]
            
            # images = [Image.open(i) for i in image_path]
            images = [load_image(img) for img in img_paths]
            inputs = processor(text=tags, 
                images=images, 
                return_tensors="pt", 
                padding=True).to(device)

            labels = torch.tensor(labels)            
            logits = model(inputs)
           
            # 模型预测
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')  # or 'binary' if binary classification
    print(f"Accuracy: {acc:.4f}, Recall: {recall:.4f}")
    report = classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=["Q-1", "Q0", "Q2"], digits=4)
    print(report)

    # return acc, recall

def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 CLIP 模型和处理器
    model_name = "chinese-clip-vit-base-patch16"

    clip_model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)

    checkpoint = "./models_chinese_3cls/clip_finetuned_classifier_epoch4.pth"
    # checkpoint = None

    # 构造数据集，假设数据目录为 "dataset"
    dataset = CustomImageDataset(file_path='datasets/fine/index_eval.jsonl', transform=None)
    # dataset = CustomImageDataset(file_path='datasets/oversea/index_train_0503.jsonl', transform=None)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    
    num_classes = len(dataset.classes)
    model = CLIPClassifier(clip_model, num_classes)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)

    evaluate(model, dataloader, processor, device)


if __name__ == "__main__":
    main()
