import pandas as pd
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import os
import gc
import h5py

class VideoAuditDataset:
    def __init__(self, src_data, online_features, offline_dir, set_name, label_name="label"):
        self.set_name = set_name
        self.online_input = src_data.loc[:, online_features]
        self.labels = src_data.loc[:, [label_name]]
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self._build_offline_input(src_data, offline_dir)
        self.offline_input = h5py.File(f'{offline_dir}/{self.set_name}.hdf5')

    def _build_offline_input(self, src_data, offline_dir):
        dropped = []
        with h5py.File(f'{offline_dir}/{self.set_name}.hdf5', 'w') as f:
            for i, image_path in enumerate(tqdm(src_data["cover_path"], desc="Offloading images")):
                try:
                    image = Image.open(image_path)
                    image_arr = np.asarray(self.transform(image)) # (3, H, W)
                    f.create_dataset(image_path, data=image_arr)
                except Exception as e:
                    dropped.append(i)
                    tqdm.write(f"Dropping {i}-th row: {e}")
                    continue
        self.online_input.drop(dropped, axis=0, inplace=True)
        self.online_input.reset_index(drop=True, inplace=True)
        self.labels.drop(dropped, axis=0, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.online_input)

    def __getitem__(self, index):
        if isinstance(index, slice):
            meta = self.online_input.iloc[index.start : index.stop : index.step]
            labels = self.labels.iloc[index.start : index.stop : index.step]
            image_paths = meta["cover_path"]
            images = []
            for image_path in image_paths:
                images.append(self.offline_input[image_path][()])
            images = np.stack(images, axis=0) # (B, 3, H, W)
        elif isinstance(index, np.ndarray):
            meta = self.online_input.iloc[index]
            labels = self.labels.iloc[index]
            image_paths = meta["cover_path"]
            images = []
            for image_path in image_paths:
                images.append(self.offline_input[image_path][()])
            images = np.stack(images, axis=0) # (B, 3, H, W)
        else:
            images = self.offline_input["pixels"][index]
            meta = self.online_input.iloc[index]
            labels = self.labels.iloc[index]
            image_path = meta["cover_path"][0]
            images = self.offline_input[image_path][()]
        return images, meta, labels