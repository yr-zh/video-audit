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
from utils import *

class BaseTorchClass(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.device = os.getenv("TORCH_DEVICE", "cuda")
        else:
            self.device = "cpu"

class FeatureEmbedding(BaseTorchClass):
    def __init__(self, dataframe, binned_feat_names, sparse_feat_names, embedding_dim, max_cls_map, bins_map):
        super().__init__()
        self.class_index_maps = {}
        self.class_bins_maps = {}
        self.embedding_dict = nn.ModuleDict()
        self.binned_feat_names = binned_feat_names
        self.sparse_feat_names = sparse_feat_names
        self.embedding_dim = embedding_dim
        self.max_cls_map = max_cls_map
        self.bins_map = bins_map
        self._prepare(dataframe)
        for key in self.embedding_dict.keys():
            nn.init.uniform_(
                self.embedding_dict[key].weight,
                a=-0.01, 
                b=0.01
            )
        self.to(self.device)
        
    def _classify_sparse_feature(self, dataframe, feat_name, max_cls=5000):
        '''Classify the values in the `feat_name` column into (at most `max_cls`) groups'''
        max_cls = self.max_cls_map.get(feat_name, max_cls)
        most_common_values = dataframe[feat_name].value_counts().head(max_cls).index
        class_index_dict = {k: v for v, k in enumerate(most_common_values) if not k == "未知"}
        self.class_index_maps[feat_name] = class_index_dict

    def _classify_binned_feature(self, dataframe, feat_name, max_cls=100):
        '''Classify the values in the `feat_name` column into (at most `max_cls`) groups'''
        max_cls = self.max_cls_map.get(feat_name, max_cls)
        _, bins = pd.qcut(dataframe[feat_name], max_cls, duplicates="drop", labels=False, retbins=True)
        bins = bins.tolist()
        bins = self.bins_map.get(feat_name, bins)
        if not len(bins) == 1:
            bins[0] = -float("inf")
            bins[-1] = float("inf")
        else:
            bins = [-float("inf"), float("inf")]
        self.class_bins_maps[feat_name] = bins

    def _create_embedding_dict(self, feat_name, embedding_dim):
        if feat_name in self.sparse_feat_names:
            self.embedding_dict[feat_name] = nn.Embedding(
                len(self.class_index_maps[feat_name]) + 1, embedding_dim
            )
        elif feat_name in self.binned_feat_names:
            n_bins = len(self.class_bins_maps[feat_name]) - 1
            self.embedding_dict[feat_name] = nn.Embedding(
                n_bins, embedding_dim
            )
        else:
            raise Exception(f"Unknown feature name: {feat_name}")

    def _prepare(self, dataframe):
        print("Preparing sparse features ...")
        for name in self.sparse_feat_names:
            self._classify_sparse_feature(dataframe, name)
            self._create_embedding_dict(name, self.embedding_dim)
        print("Preparing binned features ...")
        for name in self.binned_feat_names:
            self._classify_binned_feature(dataframe, name)
            self._create_embedding_dict(name, self.embedding_dim)

    def _get_class_indices(self, dataframe, feat_name):
        if feat_name in self.sparse_feat_names:
            cls_names = dataframe[feat_name]
            class_index_dict = self.class_index_maps[feat_name]
            cls_idx = [class_index_dict.get(name, len(class_index_dict)) for name in cls_names]
        elif feat_name in self.binned_feat_names:
            try:
                binned = pd.cut(
                    dataframe[feat_name], 
                    self.class_bins_maps[feat_name],
                    labels=False
                )
                cls_idx = binned.astype(int).to_list()
            except:
                print(feat_name)
                print(dataframe[feat_name])
                print(self.class_bins_maps[feat_name])
                print(binned.to_list())
                raise Exception
        else:
            raise Exception(f"Unknown feature name: {feat_name}")
        return cls_idx

    def _get_latent_vectors(self, dataframe, feat_name):
        class_indices = self._get_class_indices(dataframe, feat_name)
        return self.embedding_dict[feat_name](
            torch.LongTensor(class_indices).to(self.device) # loading to self.device
        )

    def forward(self, dataframe):
        all_feat_names = self.sparse_feat_names + self.binned_feat_names
        e = [
            self._get_latent_vectors(dataframe, name) 
            for name in all_feat_names
        ]
        return torch.cat(e, dim=1) # B E * n_feats

class ResNet(BaseTorchClass):
    def __init__(self, num_blocks, embedding_dim):
        super().__init__()
        assert num_blocks in [18, 34, 50, 101, 152]
        self.net = getattr(tv.models, f"resnet{num_blocks}")()
        fc_in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(
            in_features=fc_in_features,
            out_features=embedding_dim
        )
        self.to(self.device)

    def forward(self, x):
        return self.net(x) # B 3 H W -> B embedding_dim

class MLP(BaseTorchClass):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        num_layers=3, 
        batch_normalization=True,
        dropout_p=0.5
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(in_dim,hidden_dim)
        self.dropout_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.dropout_layers.append(
                nn.Dropout(p=dropout_p)
            )
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(
                nn.Linear(hidden_dim,hidden_dim)
            )
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.bn_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
        self.output_layer = nn.Linear(hidden_dim,out_dim)
        self.fn = nn.ReLU()
        self.init_modules()
        self.to(self.device)

    def init_modules(self):
        nn.init.kaiming_normal_(self.input_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)
        for layer in self.hidden_layers:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        x = self.fn(self.input_layer(x))
        for dp, fc, bn in zip(
            self.dropout_layers, self.hidden_layers, self.bn_layers
        ):
            x = self.fn(bn(fc(dp(x))))
        x = self.output_layer(x)
        return x

class ResNetAuditor(BaseTorchClass):
    def __init__(
        self, 
        dataframe, 
        binned_features, sparse_features, dense_features,
        num_residual_blocks=18, resnet_embedding_dim=128,
        feature_embedding_dim=128,
        mlp_hidden_dim=128, num_mlp_layers=4,
        out_dim=4,
        max_cls_map={},
        bins_map={}
    ):
        super().__init__()
        self.binned_features = binned_features
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.resnet = ResNet(num_residual_blocks, resnet_embedding_dim)
        if self.binned_features or self.sparse_features:
            self.embedding = FeatureEmbedding(
                dataframe, 
                binned_features, sparse_features, 
                feature_embedding_dim,
                max_cls_map=max_cls_map,
                bins_map=bins_map
            )
        mlp_in_dim = (
            len(self.sparse_features) + len(self.binned_features)
        ) * feature_embedding_dim + len(self.dense_features) + resnet_embedding_dim
        self.classifier = MLP(
            in_dim=mlp_in_dim, 
            out_dim=out_dim, 
            hidden_dim=mlp_hidden_dim,
            num_layers=num_mlp_layers
        )

        self.logits2prob_fn = nn.Sigmoid()

    def _gen_classifier_input(self, image_data, meta_data):
        resnet_embedding = self.resnet(
            torch.from_numpy(
                image_data
            ).float().to(self.device)
        ) # B 3 H W -> B resnet_embedding_dim
        if self.binned_features or self.sparse_features:
            feature_embedding = self.embedding(meta_data) # (B, n_sparse_feats * embedding_dim)
            inp = torch.cat(
                [
                    resnet_embedding,
                    feature_embedding,
                    torch.from_numpy(
                        meta_data[self.dense_features].values
                    ).float().to(self.device)
                ],
                dim=1
            ) # (B, resnet_embedding_dim + (n_sparse_feats + n_binned_feats) * feature_embedding_dim + n_dense_feats)
        else:
            inp = resnet_embedding
        return inp

    def forward(self, image_data, meta_data):
        classifier_input = self._gen_classifier_input(image_data, meta_data)
        output = self.classifier(classifier_input) # (B, 4)
        return output

    def predict(self, image_data, meta_data):
        classifier_output = self.forward(image_data, meta_data)
        return self.logits2prob_fn(classifier_output)

    def fit(
        self, 
        trainset, testset, 
        optimizer,  
        loss_fn, 
        save_dir,
        num_epochs=10, 
        batch_size=128, 
        log_interval=5, 
        eval_interval=50,
        report_to="console",
        log_dir="logs"
    ):
        tqdm.write("Training data example:")
        tqdm.write(trainset.online_input.head(1).to_string(index=False, justify='right'))
        tqdm.write("Testing data example:")
        tqdm.write(testset.online_input.head(1).to_string(index=False, justify='right'))
        assert report_to in ["console", "tensorboard"], "`report_to` must be `console`/`tensorboard`"
        reporter = Reporter(report_to, log_dir)
        final_model_path = save_dir + os.sep + "model.pt"
        earlystopping = EarlyStopping(
            final_model_path,
            mode="min"
        )
        step = 0
        num_batches = math.ceil(len(trainset) / batch_size)
        for epoch in trange(num_epochs, desc="Epochs", position=0):
            start_idx = 0
            index_pool = np.random.permutation(len(trainset))
            total_iter = math.ceil(len(index_pool) / batch_size)
            iter_bar = tqdm(total=total_iter, desc="Iterations", position=1)
            while start_idx < len(index_pool):
                indices = index_pool[start_idx : start_idx + batch_size]
                start_idx += batch_size
                batch_images, batch_meta, batch_labels = trainset[indices]
                optimizer.zero_grad()
                predictions = self(batch_images, batch_meta) # (B, C)
                loss = loss_fn(
                    predictions,  # (B, C)
                    torch.from_numpy(
                        batch_labels.values
                    ).float().to(self.device) # (B, C)
                )
                loss.backward()
                optimizer.step()
                if step % log_interval == 0:
                    reporter.display(
                        stage="training",
                        step=step,
                        value_dict={
                            f"loss": loss.item()
                        }
                    )
                if step % eval_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        batch_images, batch_meta, batch_labels = testset[np.arange(len(testset))]
                        TP, TN, FP, FN = evaluate_model(
                            self, 
                            batch_images, 
                            batch_meta,
                            batch_labels
                        )
                        tqdm.write(f"Num. positive samples: {TP + FN}")
                        tqdm.write(f"Num. negative samples: {TN + FP}")
                        underkill_rate = FN / (TP + FN) if (TP + FN) != 0 else 0
                        overkill_rate = FP / (TN + FP) if (TN + FP) != 0 else 0
                        valid_underkill_rate = underkill_rate + (1 - underkill_rate) / (overkill_rate - 0.15) * max(0, overkill_rate - 0.15)
                        reporter.display(
                            stage="validation",
                            step=step,
                            value_dict={
                                "underkill_rate": underkill_rate,
                                "overkill_rate": overkill_rate,
                                "valid_underkill_rate": valid_underkill_rate,
                                "TP": TP,
                                "TN": TN,
                                # "FP": FP,
                                # "FN": FN,
                            }
                        )
                        earlystopping(
                            valid_underkill_rate,
                            self
                        )
                    self.train()
                    # if earlystopping.earlystop:
                    #     return
                step += 1
                iter_bar.update(1)
            iter_bar.close()

class NSFWAuditor(ResNetAuditor):
    def __init__(
        self, 
        dataframe, 
        binned_features, sparse_features, dense_features,
        backbone_path,
        resnet_embedding_dim=128,
        feature_embedding_dim=128,
        mlp_hidden_dim=128, num_mlp_layers=4,
        out_dim=4,
        max_cls_map={},
        bins_map={}
    ):
        super().__init__(
            dataframe, 
            binned_features, sparse_features, dense_features,
            num_residual_blocks=50, resnet_embedding_dim=resnet_embedding_dim,
            feature_embedding_dim=128,
            mlp_hidden_dim=128, num_mlp_layers=4,
            out_dim=out_dim,
            max_cls_map={},
            bins_map={}
        )
        self.resnet = tv.models.resnet50()
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )
        self.resnet.load_state_dict(torch.load(backbone_path))
        self.resnet.fc = nn.Linear(2048, resnet_embedding_dim)
        # for name, param in self.resnet.named_parameters():
        #     if 'fc' not in name:
        #         param.requires_grad = False
        if self.binned_features or self.sparse_features:
            self.embedding = FeatureEmbedding(
                dataframe, 
                binned_features, sparse_features, 
                feature_embedding_dim,
                max_cls_map=max_cls_map,
                bins_map=bins_map
            )
        mlp_in_dim = (
            len(self.sparse_features) + len(self.binned_features)
        ) * feature_embedding_dim + len(self.dense_features) + resnet_embedding_dim
        self.classifier = MLP(
            in_dim=mlp_in_dim, 
            out_dim=out_dim, 
            hidden_dim=mlp_hidden_dim,
            num_layers=num_mlp_layers
        )
        self.to(self.device)

    def predict(self, image_data, meta_data):
        classifier_output = self.forward(image_data, meta_data) # (B, C)
        _, predicted_class = torch.max(classifier_output, dim=1)
        return predicted_class # (B, )

    def fit(
        self, 
        trainset, testset, 
        optimizer,  
        loss_fn, 
        save_dir,
        num_epochs=10, 
        batch_size=128, 
        log_interval=5, 
        eval_interval=50,
        report_to="console",
        log_dir="logs"
    ):
        tqdm.write("Training data example:")
        tqdm.write(trainset.online_input.head(1).to_string(index=False, justify='right'))
        tqdm.write("Testing data example:")
        tqdm.write(testset.online_input.head(1).to_string(index=False, justify='right'))
        assert report_to in ["console", "tensorboard"], "`report_to` must be `console`/`tensorboard`"
        reporter = Reporter(report_to, log_dir)
        final_model_path = save_dir + os.sep + "model.pt"
        earlystopping = EarlyStopping(
            final_model_path,
            mode="min"
        )
        step = 0
        num_batches = math.ceil(len(trainset) / batch_size)
        for epoch in trange(num_epochs, desc="Epochs", position=0):
            start_idx = 0
            index_pool = np.random.permutation(len(trainset))
            total_iter = math.ceil(len(index_pool) / batch_size)
            iter_bar = tqdm(total=total_iter, desc="Iterations", position=1)
            while start_idx < len(index_pool):
                indices = index_pool[start_idx : start_idx + batch_size]
                start_idx += batch_size
                batch_images, batch_meta, batch_labels = trainset[indices]
                optimizer.zero_grad()
                predictions = self(batch_images, batch_meta) # (B, C)
                loss = loss_fn(
                    predictions,  # (B, C)
                    torch.from_numpy(
                        batch_labels.values.flatten()
                    ).long().to(self.device) # (B, )
                )
                loss.backward()
                optimizer.step()
                if step % log_interval == 0:
                    reporter.display(
                        stage="training",
                        step=step,
                        value_dict={
                            f"loss": loss.item()
                        }
                    )
                if step % eval_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        batch_images, batch_meta, batch_labels = testset[np.arange(len(testset))]
                        (
                            precision_class_1, 
                            precision_class_2, 
                            precision_class_3, 
                            underkill_rate, 
                            overkill_rate
                        ) = evaluate_model(
                            self, 
                            batch_images, 
                            batch_meta,
                            batch_labels
                        )
                        valid_underkill_rate = underkill_rate + (1 - underkill_rate) / (overkill_rate - 0.15) * max(0, overkill_rate - 0.15)
                        reporter.display(
                            stage="validation",
                            step=step,
                            value_dict={
                                "underkill_rate": underkill_rate,
                                "overkill_rate": overkill_rate,
                                "valid_underkill_rate": valid_underkill_rate,
                                "prec_cls_1": precision_class_1,
                                "prec_cls_2": precision_class_2,
                                "prec_cls_3": precision_class_3,
                            }
                        )
                        earlystopping(
                            valid_underkill_rate,
                            self
                        )
                    self.train()
                    # if earlystopping.earlystop:
                    #     return
                step += 1
                iter_bar.update(1)
            iter_bar.close()


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