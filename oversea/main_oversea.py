import sys
import os
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
from models import *
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from models import CLIPClassifier, CLIPClassifierOversea
from sentence_transformers import SentenceTransformer, util


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__file__)

sut_port = int(os.getenv("SUT_PORT", 80))
model_path = os.getenv("MODEL_PATH", "model/clip_finetuned_classifier_epoch2.pth")
black_list_path = os.getenv("BLACK_LIST", "")
white_list_path = os.getenv("WHITE_LIST", "")

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
# load model
    
model_name = "model/chinese-clip-vit-base-patch16"

clip_model = ChineseCLIPModel.from_pretrained(model_name)
processor = ChineseCLIPProcessor.from_pretrained(model_name)

img_model = SentenceTransformer('model/clip-ViT-B-32')
text_model = SentenceTransformer('model/clip-ViT-B-32-multilingual-v1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# finetune_model = 'model/clip_finetuned_classifier_epoch2.pth'
finetune_model = model_path
model = CLIPClassifierOversea(clip_model, text_model, img_model, 2)
model.load_state_dict(torch.load(finetune_model))
model.to(device)
model = model.eval()
logger.info(f"load model: {finetune_model}")
app_ready = True

# black_list_path = "./blacklist.txt"
if black_list_path:
    with open(black_list_path, 'r', encoding="utf-8") as f:
        black_list = [line.strip() for line in f]
else:
    black_list = []

# white_list_path = "./white_list.txt"
if white_list_path:
    with open(white_list_path, 'r', encoding="utf-8") as f:
        white_list = [line.strip() for line in f]
else:
    white_list = []

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
    global train_task_proc
    req = request.get_json()
    return {'success': True, 'msg': ''}

@app.route('/v1/multiple_classification/train_status', methods=['GET'])
def train_status():
    return {'success': True, 'status': 'SUCCESS', 'msg': ''}

@app.route('/v1/multiple_classification/load_predict', methods=['POST'])
def load_predict():
    global transformations, model
    req = request.get_json()
    return {'success': True, 'msg': ''}

@app.route('/v1/multiple_classification/predict_status', methods=['GET'])
def predict_status():
    return {'success': True, 'msg': '', 'status': 'READY'}

@app.route('/v1/multiple_classification/predict', methods=['POST'])
def predict():
    global transformations, model
    req = request.get_json()
    # item_meta = req["content"]["meta"]
    content = req["content"]["meta"]["title"]
    tag = req["content"]["meta"]["tag"]
    blogger_name = req["content"]["meta"]["bloggerName"]

    if black_list and blogger_name in black_list:
        print(f"the blogger {blogger_name} in blacklist")
        return {'success': True, 'msg': '', 'pred_label': 'Q0'}

    if white_list and blogger_name in white_list:
        print(f"the blogger {blogger_name} in whitelist")
        return {'success': True, 'msg': '', 'pred_label': 'Q2'}

    cover_fp = io.BytesIO(base64.b64decode(req["content"]["cover"]))
    cover_img = Image.open(cover_fp)

    inputs = processor(text=tag, 
        images=cover_img, 
        return_tensors="pt", 
        padding=True).to(device)

    outputs = model(content, cover_img, inputs)
    predicted_cls = int(torch.argmax(outputs))

    print(outputs)
    print(torch.argmax(outputs))

    label_map = {
        0: "Q0",
        1: "Q2"
    }

    return {'success': True, 'msg': '', 'pred_label': label_map[predicted_cls]}

if __name__ == '__main__':
    app.run("0.0.0.0", sut_port)

