import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from dataset.bdd100k import BDD100KDataset
from utils.preprocess import minmax_normalize

import datetime
from torchsummary import summary

##################################
import albumentations as albu
import cv2
import argparse
resizer = albu.Compose([albu.Resize(height=256, width=256)])
def arg_parse():
    """
    Parse argumentets to segmentation module
    """
    parser = argparse.ArgumentParser(description='Freespace detection module')
    parser.add_argument("--bs", dest = "batch_size", default = 1, type = int)
    parser.add_argument("--video", dest = "videofile", help = "video file tor run segmentation on", default = "test.mp4")
    parser.add_argument("--dim", dest = "input_dim", help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = 256, type = int)
    return parser.parse_args()

args = arg_parse()
##################################

print("Loading network...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SPPNet(output_channels=3, enc_type='mobilenetv2', dec_type='maspp').to(device)
model_path = '../model/bdd100k_deeplabv3p_mobilenetv2/model_tmp.pth'
print("Setting the parameters")
param = torch.load(model_path)
model.load_state_dict(param)
del param
print("Network is ready")

batch_size = int(args.batch_size)

#valid_dataset = BDD100KDataset(split='valid', net_type='deeplab')
#valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

"""images_list = []
labels_list = []
preds_list = []"""

# Set up the model in evaluation mode
model.eval()

videofile = args.videofile
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), "Cannot capture source"

palette = np.array([[0, 0, 0],     # black
                     [0, 0, 255],   # red in BGR color system
                     [255, 0, 0],  # blue in BGR color system
                     [0, 255, 255]]) # yellow in BGR color system

frames = 0
start = datetime.datetime.now()
delta_list = []
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        #cv2.imshow("frame", frame)
        with torch.no_grad():
            """
            What you need to do is convert the opencv frame to a pytorch tensor. You can do it as follows:
            t = 1 - torch.tensor(np.ascontiguousarray(np.flip(im, 2)).transpose(2,0,1))

            In OpenCV images are stored like (height, width, channels) whereas in pytorch you expect (channels, height, width) (thats the transpose part). Further OpenCV stores as BGR instead of RGB (therefore the flip). Making the array contiguous is necessary because the flip actually does not change the underlying data at all and only modifies the iterator (which seems to not work when further transposing). Also in the small test that I did OpenCV loaded a uint8 image for me
            which is also reversed (0 black vs. 0 white), therefore the 1 - torch.tensor(...)

            Edit: Oh yeah, if you want a fastai image, you will then also need to do Image(t)
            """
            resized = resizer(image=frame)
            frame = resized['image']
            #print(frame)
            img = np.array(frame)
            # Transformation into numpy array
            #img = np.ascontiguousarray(frame)
            #img = np.ascontiguousarray(np.flip(frame, 2))
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #img = np.array(img)
            #cv2.imshow('img', img)
            img = minmax_normalize(img, norm_range=(-1, 1)) 
            img = img.transpose(2, 0, 1) # OpenCV (256, 256, 3) -> PyTorch (3, 256, 256)
            img = [img] # One sample in a batch
            image = torch.FloatTensor(img)
            #image = torch.Tensor(img)
            #print("CV Mat", image.shape, image) 
            image = image.to(device)

            prev = datetime.datetime.now()
            # prediction
            pred = model.tta(image, net_type='deeplab')
            #print("pred shape", pred.shape) # torch sieze
            pred = pred.argmax(dim=1)
            #print("pred shape", pred.shape) # torch sieze
            pred_np = pred.detach().cpu().numpy()
            cur = datetime.datetime.now()
            cv2.imshow('frame', frame) 
            #print("pred unique", np.unique(pred_np))
            #print("pred", pred_np.shape, pred) 
            #print(palette[pred_np[0]]) 
            pred_lbl = np.array(palette[pred_np[0]], np.uint8)
            #print("pred_lbl", pred_lbl.shape, pred_lbl)
            cv2.imshow("pred", pred_lbl)
            
            #print("frame", frame.shape, "pred_lbl", pred_lbl.shape)
            blend = np.bitwise_or(frame, pred_lbl)
            #blend  = np.clip(np.add(frame, pred_lbl), 0, 255)
            #print("blend", blend.shape, blend)
            #print("pred_lbl", pred_lbl.shape, pred_lbl)
            cv2.imshow("blend", blend)
            #preds_np = cv2.integral(np.preds_np[0])
            #print("pred", preds_np.shape, "dtype", preds_np.dtype)
            #cv2.imshow("pred", preds_np)
            
            diff = cur - prev
            print("Time", diff)
            delta_list.append(diff)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            delta_sum = sum(delta_list, datetime.timedelta(0))
            print("*********")
            print("Sum", delta_sum)
            print("Mean", delta_sum/len(delta_list))
            break
        continue
    else:
        break
delta_sum = sum(delta_list, datetime.timedelta(0))
print("*********")
print("Sum", delta_sum)
print("Mean", delta_sum/len(delta_list))
"""with torch.no_grad():
    for batched in valid_loader:
        images, labels = batched
        images_np = images.numpy().transpose(0, 2, 3, 1)
        labels_np = labels.numpy()

        images, labels = images.to(device), labels.to(device)
        prev = datetime.datetime.now()
        preds = model.tta(images, net_type='deeplab')
        preds_np = preds.detach().cpu().numpy()
        #print("Test time augmentation", preds_np.shape)
        #print(preds_np)
        preds = preds.argmax(dim=1)
        preds_np = preds.detach().cpu().numpy()
        #print("Arg max : preds", preds.shape)# (1, 128, 256) [batch : 1][row : r][col : c]
        #print(preds_np)
        #print("Pred", np.unique(preds_np)) # [0] or [0, 1] or [0, 2] or [0, 1, 2]
        #print("Label", np.unique(labels_np)) # [0] or [0, 1] or [0, 2] or [0, 1, 2]
        cur = datetime.datetime.now()
        print("time : ", cur - prev)
        images_list.append(images_np)
        labels_list.append(labels_np)
        preds_list.append(preds_np)
        
        if len(images_list) == 20:
            break

images = np.concatenate(images_list)
labels = np.concatenate(labels_list)
preds = np.concatenate(preds_list)

# Ignore index
ignore_pixel = labels == 255
preds[ignore_pixel] = 0
labels[ignore_pixel] = 0

# Plot
fig, axes = plt.subplots(8, 3, figsize=(12, 10))
plt.tight_layout()

axes[0, 0].set_title('input image')
axes[0, 1].set_title('prediction')
axes[0, 2].set_title('ground truth')

palette = np.array([[0, 0, 0],     # black
                     [255, 0, 0],   # red
                     [0, 0, 255]])  # blue
for ax, img, lbl, pred in zip(axes, images, labels, preds):
    ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
    pred = palette[pred]
    ax[1].imshow(pred)
    lbl = palette[lbl]
    ax[2].imshow(lbl)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

plt.savefig('eval.png')
plt.close()
"""
