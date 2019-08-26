import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import SPPNet
from dataset.bdd100k2 import BDD100K2Dataset
from utils.preprocess import minmax_normalize

import albumentations as albu
import cv2
import datetime
import sys

import argparse
parser = argparse.ArgumentParser(description='Freespace detection module')
parser.add_argument("video_path", help = "video file tor run segmentation on" )
parser.add_argument("--bs", dest = "batch_size", default = 1, type = int, help="number of sample per batch")
parser.add_argument("--dim", dest = "input_dim", help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = 256, type = int)
parser.add_argument("--dev", dest = "device_id", help = "ID of graphic card device", default = '0')
args = parser.parse_args()

print("Loading network...")
device = torch.device('cuda:'+args.device_id if torch.cuda.is_available() else 'cpu')
model = SPPNet(output_channels=4, enc_type='mobilenetv2', dec_type='maspp').to(device)
model_path = '../model/bdd100k_deeplabv3p_mobilenetv2_lane/model_tmp.pth'
print("Setting the parameters...")
param = torch.load(model_path)
model.load_state_dict(param)
del param
print("Setting hyper parameter...")
batch_size = int(args.batch_size)
dim = int(args.input_dim)
resizer = albu.Compose([albu.Resize(height=dim, width=dim)])
print("Setting up the model in evalutation mode...")
model.eval()

print("Loading video...")
videofile = args.video_path
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), "Cannot capture source"

# Conversion of predefined labels intto colors
# keep in mind color space is BGR, it uses default opencv setup
# definition of labels is in the dataset/bdd100k2.py
palette = np.array([[0, 0, 0],     # black for backgroudn
                     [0, 0, 255],   # red for directly drivable area
                     [255, 0, 0],  # blue for alternatively drivable area
                     [255, 0, 255]]) # purple for car lanes

frames = 0
start = datetime.datetime.now()
delta_list = []
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        #cv2.imshow("frame", frame)
        with torch.no_grad():   # no backward pass, meaning no need to learn 
                                # anything, but only prediction required

            # Resizing image into 256x256, hence accelerate throughput
            resized = resizer(image=frame)
            frame = resized['image']
            #print(frame)
           
            # Transformation of an OpenCV image into a PyTorch tensor
            img = np.array(frame)
            img = minmax_normalize(img, norm_range=(-1, 1))  # network internal weights range from [-1, 1]
            img = img.transpose(2, 0, 1)    # The transpose part
                                            # OpenCV images (hegight, width, channels)
                                            # PyTorch tensor (channels, height, width)
                                            # in this case (256, 256, 3) -> (3, 256, 256)
            img = [img]                     # Specify the batch size 
                                            # For test, one sample in a batch 
                                            # (3, 256, 256) -> (1, 3, 256, 256)
            image = torch.FloatTensor(img) # tensor of float data type 
            image = image.to(device) # let gpu calculate afterwards
            
            # Inference time measure 
            prev = datetime.datetime.now()

            # Prediction using DNN (deeplabv3 plus)
            pred = model.tta(image, net_type='deeplab') # forward pass (network predction)
            pred = pred.argmax(dim=1) # best among N images where N is the size of output channel
            pred_np = pred.detach().cpu().numpy() # let cpu get the prediction


            # Inference time record
            cur = datetime.datetime.now()
            
            # Debug image visualization
            #cv2.imshow('frame', frame)
            pred_lbl = np.array(palette[pred_np[0]], np.uint8)
            #cv2.imshow("pred", pred_lbl)
            blend = np.bitwise_or(frame, pred_lbl)
            cv2.imshow("blend", blend)
            
            # Statistics
            diff = cur - prev
            print("Time", diff)
            delta_list.append(diff) # Accumulate inference delays
                                    # Inference = resizng + transformation + prediction
                                    # Optimization for each submodule needed.
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            delta_list = delta_list[1:]
            delta_sum = sum(delta_list, datetime.timedelta(0))
            print("*********")
            print("Sum", delta_sum)
            print("Mean", delta_sum/len(delta_list))
            sys.exit(0)
        continue
    else:
        break
delta_list = delta_list[1:]
delta_sum = sum(delta_list, datetime.timedelta(0))
print("*********")
print("Sum", delta_sum)
print("Mean", delta_sum/len(delta_list))
