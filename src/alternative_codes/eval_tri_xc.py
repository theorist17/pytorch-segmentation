import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from utils.preprocess import minmax_normalize

import datetime
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SPPNet(output_channels=19).to(device)
#model = SPPNet(output_channels=19, enc_type='mobilenetv2', dec_type='maspp').to(device)
summary(model, (3, 128, 256))
model_path = '../model/cityscapes_deeplab_v3_plus/model.pth'
#model_path = '../model/cityscapes_mobilenetv2/model.pth'
#model_path = '../model/cityscapes_deeplabv3p_mobilenetv2/model_tmp.pth'
param = torch.load(model_path)
model.load_state_dict(param)
del param

batch_size = 1

valid_dataset = CityscapesDataset(split='valid', net_type='deeplab')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

images_list = []
labels_list = []
preds_list = []

model.eval()
with torch.no_grad():
    print("Begininng first batch")
    prev = datetime.datetime.now()
    for batched in valid_loader:
        images, labels = batched
        images_np = images.numpy().transpose(0, 2, 3, 1)
        labels_np = labels.numpy()

        images, labels = images.to(device), labels.to(device)
        preds = model.tta(images, net_type='deeplab')
        preds = preds.argmax(dim=1)
        preds_np = preds.detach().cpu().numpy()

        images_list.append(images_np)
        labels_list.append(labels_np)
        preds_list.append(preds_np)
        
        cur = datetime.datetime.now()
        print(cur - prev)
        prev = cur
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
fig, axes = plt.subplots(4, 3, figsize=(12, 10))
plt.tight_layout()

axes[0, 0].set_title('input image')
axes[0, 1].set_title('prediction')
axes[0, 2].set_title('ground truth')

for ax, img, lbl, pred in zip(axes, images, labels, preds):
    ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
    ax[1].imshow(pred)
    ax[2].imshow(lbl)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

plt.savefig('eval.png')
plt.close()

