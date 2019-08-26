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
from dataset.avm import AVMDataset
from utils.preprocess import minmax_normalize

import datetime
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = SPPNet(output_channels=19).to(device)
model = SPPNet(output_channels=3, enc_type='mobilenetv2', dec_type='maspp').to(device)
summary(model, (3, 128, 256))
#model_path = '../model/cityscapes_deeplab_v3_plus/model.pth'
#model_path = '../model/cityscapes_mobilenetv2/model.pth'
#model_path = '../model/cityscapes_deeplabv3p_mobilenetv2/model_tmp.pth'
model_path = '../model/bdd100k_deeplabv3p_mobilenetv2/model_tmp.pth'
#model_path = '../model/avm_deeplabv3p_mobilenetv2/model_tmp.pth'
#model_path = '../model/cityscapes_deeplabv3p_double_epoch/model.pth'
param = torch.load(model_path)
model.load_state_dict(param)
del param

batch_size = 1

valid_dataset = BDD100KDataset(split='valid', net_type='deeplab')
#valid_dataset = AVMDataset(split='valid', net_type='deeplab')
#valid_dataset = CityscapesDataset(split='valid', net_type='deeplab')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

images_list = []
labels_list = []
preds_list = []

model.eval()
with torch.no_grad():
    for batched in valid_loader:
        images, labels = batched
        images_np = images.numpy().transpose(0, 2, 3, 1)
        #print(images_np.shape, images_np)
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
print(np.unique(labels))
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
'''
palette = np.array([[0, 0, 255],        # blue free space 
                     [255, 255, 255],   # white marker
                     [255, 0, 0],       # red vehicle
                     [0, 255, 0],        # green objects 
                     [0, 0, 0] ])       # black null 
'''
for ax, img, lbl, pred in zip(axes, images, labels, preds):
    ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
    pred = palette[pred]
    #print("eval pred", pred.shape, pred)
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
