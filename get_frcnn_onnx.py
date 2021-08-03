from trt_engine import InferEngine
import torch
import pandas as pd
import numpy as np
import argparse
import os
import torch
import torchvision
from torchvision import transforms as trn
from PIL import Image
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

tx = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, dummy_img):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True) 
    
    if model_path is not None:
        num_classes = 3
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    inp = cv2.imread(dummy_img) 
    inp = tx(inp)
    inp = inp.unsqueeze(0)
    outp = model(inp)
    
    onnx_opset_version = 11 
    onnx_out_fname = f'frcnn_opset={onnx_opset_version}.onnx'
    torch.onnx.export(model, 
            inp, 
            onnx_out_fname, 
            output_names = ['boxes', 'labels', 'scores'], 
            verbose = True, 
            export_params = True,
            opset_version = onnx_opset_version)

    print(f"Successfully exported model to {onnx_out_fname}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = None)
    parser.add_argument('--dummy_img', default = 'dummy_img.jpg')
    args = parser.parse_args()
    load_model(args.model_path, args.dummy_img)
    
