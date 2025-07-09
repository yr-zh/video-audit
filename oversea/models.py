import os
master_dir = os.path.dirname(__file__)
import sys
sys.path.append(master_dir)
import math
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import cv2
from utils import *


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
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        
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
class CLIPClassifierOversea(nn.Module):
    def __init__(self, clip_model, text_model, img_model, num_classes):
        """
        :param clip_model: 预训练的 CLIP 模型
        :param num_classes: 类别数量
        """
        super(CLIPClassifierOversea, self).__init__()
        self.in_features = 1024

        self.clip_model = clip_model
        self.text_model = text_model
        self.img_model = img_model

        # 冻结模型参数
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.img_model.parameters():
            param.requires_grad = False     
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.attention = AttentionFusion3d()

        # 假设图像编码器输出为 512 维，添加新的分类层
        # self.classifier = nn.Linear(self.in_features, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.in_features, 512),
        #     nn.BatchNorm1d(512),  # 加入 BN 层
        #     # nn.LayerNorm(512),  # 加入 LayerNorm
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.BatchNorm1d(512),  # 加入 BN 层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # 加入 BN 层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, texts, images, inputs):
                    
        # outputs = self.clip_model(**inputs)
        # tag_embedding = outputs.text_embeds
 
        # tag_embeddings = self.clip_model.get_text_features(**inputs)
        outputs = self.clip_model(**inputs)
        tag_embeddings = outputs.text_embeds
        img_embeddings = outputs.image_embeds
        # text_embeddings = torch.from_numpy(self.text_model.encode(texts)).to('cuda')
        # img_embeddings = torch.from_numpy(self.img_model.encode(images)).to('cuda')
        fused_vector_concat = torch.cat((tag_embeddings, img_embeddings), dim=1)
        # fused_vector = self.attention(img_embeddings, tag_embeddings, text_embeddings)
        logits = self.classifier(fused_vector_concat)
        return logits
        

def get_first_frame(video):
    # 加载视频文件
    cap = cv2.VideoCapture(video)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("无法打开视频文件")
    else:
        # 读取第一帧
        ret, frame = cap.read()
    
    # 释放资源
    cap.release()
    return frame