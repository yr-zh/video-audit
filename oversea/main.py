import sys
import os
sut_port = int(os.getenv("SUT_PORT", 8030))
import json
import logging
import subprocess
import requests
import io
import random
import pickle
import base64
from flask import Flask, request, redirect, abort
from typing import Dict
import numpy as np
import torch
torch.set_num_threads(4)
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from preprocess import *


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__file__)

app = Flask(__name__)
app_ready = False

label_list = None
train_task_proc = None
transformations = transforms.Compose(
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
model = torch.load(f"weights/model.pt")
model = model.eval()
app_ready = True

@app.route("/ready", methods=["GET"])
def ready() -> Dict[str, bool]:
    if app_ready:
        return {"ready": True}
    else:
        abort(503)

@app.route('/v1/multiple_classification/init', methods=['POST'])
def init():
    global label_list
    req = request.get_json()
    label_list = req['label_list']
    return {'success': True, 'msg': ''}

@app.route('/v1/multiple_classification/train', methods=['POST'])
def train():
    return {'success': True, 'msg': ''}

@app.route('/v1/multiple_classification/train_status', methods=['GET'])
def train_status():
    return {'success': True, 'status': 'SUCCESS', 'msg': ''}

@app.route('/v1/multiple_classification/load_predict', methods=['POST'])
def load_predict():
    return {'success': True, 'msg': ''}

@app.route('/v1/multiple_classification/predict_status', methods=['GET'])
def predict_status():
    return {'success': True, 'msg': '', 'status': 'READY'}

@app.route('/v1/multiple_classification/predict', methods=['POST'])
def predict():
    global transformations, model
    req = request.get_json()
    item_meta = req["content"]["meta"]
    cover_fp = io.BytesIO(base64.b64decode(req["content"]["cover"]))
    cover_img = Image.open(cover_fp)

    with torch.no_grad():
        image_data = transformations(cover_img).unsqueeze(0).numpy()
        meta_data = pd.DataFrame([item_meta])
        predicted_cls = model.predict(image_data, meta_data).item()
    label_map = {
        0: "Q-1",
        1: "Q0",
        2: "Q1",
        3: "Q2",
    }

    return {'success': True, 'msg': '', 'pred_label': label_map[predicted_cls]}

if __name__ == '__main__':
    app.run("0.0.0.0", sut_port)

