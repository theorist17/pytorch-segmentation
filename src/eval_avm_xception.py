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

import cv2

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
#model = SPPNet(output_channels=19).to(device)
model = SPPNet(output_channels=4, enc_type='xception65', dec_type='aspp').to(device)
summary(model, (3, 128, 256))
#model_path = '../model/cityscapes_deeplab_v3_plus/model.pth'
#model_path = '../model/cityscapes_mobilenetv2/model.pth'
#model_path = '../model/cityscapes_deeplabv3p_mobilenetv2/model_tmp.pth'
#model_path = '../model/bdd100k_deeplabv3p_mobilenetv2/model_tmp.pth'
model_path = '../model/avm_deeplabv3p_xception65/model_tmp.pth'
#model_path = '../model/cityscapes_deeplabv3p_double_epoch/model.pth'
param = torch.load(model_path)
model.load_state_dict(param)
del param

batch_size = 1

#valid_dataset = BDD100KDataset(split='valid', net_type='deeplab')
valid_dataset = AVMDataset(split='valid', net_type='deeplab')
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
#print(np.unique(labels))
#print(labels)
ignore_pixel = labels == 255
preds[ignore_pixel] = 4
labels[ignore_pixel] = 4

# Plot
fig, axes = plt.subplots(8, 3, figsize=(12, 10))
plt.tight_layout()

axes[0, 0].set_title('input image')
axes[0, 1].set_title('prediction')
axes[0, 2].set_title('ground truth')

'''
palette = np.array([[0, 0, 0],     # black
                     [255, 0, 0],   # red
                     [0, 0, 255]])  # blue
                     '''
palette = np.array([[0, 0, 255],        # blue free space 
                     [255, 255, 255],   # white marker
                     [255, 0, 0],       # red vehicle
                     [0, 255, 0],        # green objects 
                     [0, 0, 0] ])       # black null 

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

cv2.imshow("image", minmax_normalize(images[0], norm_range=(0, 1), orig_range=(-1, 1)))

palette = np.array([[255, 0, 0],        # blue free space 
                     [255, 255, 255],   # white marker
                     [0, 0, 255],       # red vehicle
                     [0, 255, 0],        # green objects 
                     [0, 0, 0] ])       # black null 
pred_np = palette[preds[0]]
pred_uint = np.array(pred_np, dtype='uint8')
h, w = pred_uint.shape[0], pred_uint.shape[1]
pred_cv = cv2.resize(pred_uint, (w, h))
cv2.imshow("pred", pred_cv)

mask = np.array([[0, 0, 0],        # blue free space 
                     [255, 255, 255],   # white marker
                     [0, 0, 0],       # red vehicle
                     [0, 0, 0],        # green objects 
                     [0, 0, 0] ])       # black null 
marker_np = mask[preds[0]]
marker_uint = np.array(marker_np, dtype='uint8')
h, w = marker_uint.shape[0], marker_uint.shape[1]
marker_cv = cv2.resize(marker_uint, (w, h))
cv2.imshow("marker", marker_cv)

corner_uint = np.array(marker_np, dtype='uint8')
h, w = corner_uint.shape[0], corner_uint.shape[1]
corner_cv = cv2.resize(corner_uint, (w, h))
gray_cv = cv2.cvtColor(corner_cv, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray_cv, 100, 0.01, 10) 
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(corner_cv, (x,y), 5, (255, 0, 255), -1)
cv2.imshow('marker & corner', corner_cv)

harris_uint = np.array(marker_np, dtype='uint8')
h, w = harris_uint.shape[0], harris_uint.shape[1]
harris_cv = cv2.resize(harris_uint, (w, h))
gray_cv = cv2.cvtColor(harris_cv, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray_cv, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
harris_cv[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('marker & harris', harris_cv)

img_uint = np.array(marker_np, dtype='uint8')
h, w = img_uint.shape[0], img_uint.shape[1]
img_cv = cv2.resize(img_uint, (w, h))
gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
#morp_cv = cv2.erode(gray_cv, kernel,iterations=1)
#morp_cv = cv2.dilate(morp_cv, kernel, iterations=1)
morp_cv = cv2.dilate(gray_cv, kernel, iterations=1)
cv2.imshow("morp", morp_cv)
blur_cv = cv2.medianBlur(morp_cv, 5)
adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresh_type = cv2.THRESH_BINARY_INV
bin_img = cv2.adaptiveThreshold(blur_cv, 255, adapt_type, thresh_type, 11, 2)
cv2.imshow("bin_img", bin_img)
rho, theta, thresh = 1, np.pi/180, 10
'''lines = cv2.HoughLines(blur_cv, rho, theta, thresh)
if lines is not None:
    for rho, theta in lines[0]:
    #for i in range(0, len(lines)):
        #rho = lines[i][0][0]
        #theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_cv, (x1, y1), (x2,y2), (0,0,255), 2)
else:
    print("No lines found")'''
lines = cv2.HoughLinesP(gray_cv, rho, theta, thresh, minLineLength=10, maxLineGap=50)
print("lines", lines)
if lines is not None:
    #for line in lines:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    print("No lines found")
cv2.imshow('marker & hough', img_cv)


from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on 
    the unit circle to segment `k` angles inside `lines`."""

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)
    
    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                                for angle in angles], dtype=np.float32)
    
    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

segmented = segment_by_angle_kmeans(lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
