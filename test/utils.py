import os, json, math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm, trange
import pandas as pd

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class Reporter:
    def __init__(self, displayer="console", log_dir=None):
        self.displayer = displayer
        if displayer == "console":
            self.logger = tqdm.write
        elif displayer == "tensorboard":
            if log_dir is None:
                raise Exception("\"log_dir\" must be given for using TensorBoard")
            log_dir = os.path.abspath(log_dir + os.sep + CURRENT_TIME)
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            self.logger = SummaryWriter(log_dir)
        else:
            raise NotImplementedError("Only `console` and `tensorboard` are supported")

    def display(self, stage, step, value_dict):
        if self.displayer == "console":
            head = stage.upper()
            tail = [f"STEP: {step}"]
            for k, v in value_dict.items():
                if k.upper() == "LOSS":
                    pair = f"{k.upper()}: {v:.4e}"
                else:
                    pair = f"{k.upper()}: {v:.3f}"
                tail.append(pair)
            content = " | " + head + " | " + ", ".join(tail)
            self.logger(content)
        elif self.displayer == "tensorboard":
            for k, v in value_dict.items():
                self.logger.add_scalar(
                    f"{stage.upper()}/{k.upper()}",
                    v,
                    step
                )

class EarlyStopping:
    def __init__(self, save_path, patience=5, delta=0, mode="min"):
        self.patience = patience
        self.delta = delta
        self.path = save_path
        assert mode in ["min", "max"], "`mode` must be either `min` or `max`"
        self.mode = mode
        self.operator = "<" if mode == "max" else ">"
        self.counter = 0
        self.best_score = None
        self.earlystop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif not self._is_improved(score):
            self.counter += 1
            tqdm.write(
                f"{score:.3e}{self.operator}{self.best_score:.3e} -> Earlystopping: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.earlystop = True
                tqdm.write("Training stopped")
        else:
            self.best_score = score
            self.save_model(model)
            self.counter = 0

    def _is_improved(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    def save_model(self, model):
        tqdm.write(f"Saving a better model to: {self.path}")
        torch.save(
            model, self.path
        )

def split_unbalanced_data(dataframe, target_neg_div_pos=3, min_size=1000):
    positive_dataframe = dataframe[dataframe["label"] == 1]
    negative_dataframe = dataframe[dataframe["label"] == 0]
    negative_partition_size = int(len(positive_dataframe) * target_neg_div_pos)
    index_pool = np.random.permutation(len(negative_dataframe))
    ret = {}
    start = 0
    i = 0
    while start < len(index_pool):
        part_indices = index_pool[start : start + negative_partition_size]
        start += negative_partition_size
        negative_part = negative_dataframe.iloc[part_indices]
        if len(negative_part) + len(positive_dataframe) >= min_size:
            ret[f"set_{i}"] = pd.concat(
                [
                    negative_part,
                    positive_dataframe
                ],
                axis=0,
                ignore_index=True
            )
        i += 1
    print(f"Negative data are split into {len(ret)} parts.")
    return ret

def evaluate_model(model, images, meta, labels):
    prediction = model.predict(images, meta).cpu().numpy() # (B,)
    labels = labels.values.flatten() # (B,)

    precision_class_1 = precision_score(labels, prediction, labels=[1], average='macro')
    precision_class_2 = precision_score(labels, prediction, labels=[2], average='macro')
    precision_class_3 = precision_score(labels, prediction, labels=[3], average='macro')
    
    underkill_mask = np.isin(prediction, [2, 3]) & np.isin(labels, [0, 1])
    underkill_rate = np.mean(underkill_mask)
    
    overkill_mask = np.isin(prediction, [0, 1]) & np.isin(labels, [2, 3])
    overkill_rate = np.mean(overkill_mask)
    return precision_class_1, precision_class_2, precision_class_3, underkill_rate, overkill_rate