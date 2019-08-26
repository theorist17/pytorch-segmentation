from functools import partial
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset
from utils.preprocess import minmax_normalize, meanstd_normalize
from utils.custum_aug import PadIfNeededRightBottom

class BDD100K2Dataset(Dataset):
    n_classes = 4
    void_classes = []
    valid_classes = [0, 1, 2, 3]
    class_map = dict(zip(valid_classes, range(n_classes)))

    def __init__(self, base_dir='../data/bdd100k', split='train',
                 affine_augmenter=None, image_augmenter=None, target_size=(1024, 1024),
                 net_type='unet', ignore_index=255, debug=False):
        #print('bdd100k2', ignore_index)
        self.debug = debug
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = 'val' if split == 'valid' else split

        self.img_paths = sorted(self.base_dir.glob(f'images/100k/{self.split}/*.jpg'))
        self.lbl_paths = sorted(self.base_dir.glob(f'drivable_lane_maps/labels/{self.split}/*.png'))
        #print('img_path len', len(self.img_paths))
        #print('lbl_path len', len(self.lbl_paths))
        
        assert len(self.img_paths) == len(self.lbl_paths)
        

        # Resize
        if isinstance(target_size, str):
            target_size = eval(target_size)
        if self.split == 'train':
            if self.net_type == 'deeplab':
                #print(target_size)
                target_size = (target_size[0] + 1, target_size[1] + 1)
            #self.resizer = albu.Compose([albu.RandomScale(scale_limit=(-0.5, 0), p=1.0),
            #                             PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1], value=0, ignore_index=self.ignore_index, p=1.0)])
                                         #albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
            self.resizer = albu.Compose([albu.Resize(height=256,width=256)])
        elif self.split == 'val':
            self.resizer = albu.Compose([albu.Resize(height=256,width=256)])
        else:
            self.resizer = None

        # Augment
        if self.split == 'train':
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path))
        if self.split == 'test':
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img)
                img = resized['image']
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            return img
        else:
            lbl_path = self.lbl_paths[index]
            lbl = np.array(Image.open(lbl_path))
            index = []
            lbl = self.encode_mask(lbl)
            # ImageAugment (RandomBrightness, AddNoise...)
            if self.image_augmenter:
                augmented = self.image_augmenter(image=img) # Apply
                img = augmented['image']
            # Resize (Scale & Pad & Crop)
            if self.net_type == 'unet':
                img = minmax_normalize(img)
                img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img, mask=lbl) # Apply_to_mask
                img, lbl = resized['image'], resized['mask']
            # AffineAugment (Horizontal Flip, Rotate...)
            if self.affine_augmenter:
                augmented = self.affine_augmenter(image=img, mask=lbl) # Apply_to_mask
                img, lbl = augmented['image'], augmented['mask']
            if self.debug:
                print(lbl_path)
                print(np.unique(lbl))
            else:
                img = img.transpose(2, 0, 1)
                img = torch.FloatTensor(img)
                lbl = torch.LongTensor(lbl)
            #print("Lbl in __get_item_ is", lbl.size())
            return img, lbl

    def encode_mask(self, lbl):
        for c in self.void_classes:
            lbl[lbl == c] = self.ignore_index
        for c in self.valid_classes:
            lbl[lbl == c] = self.class_map[c]
        return lbl


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.custum_aug import Rotate

    affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                     # Rotate(5, p=.5)
                                     ])
    # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
    #                                 albu.RandomBrightnessContrast(p=.5)])
    image_augmenter = None
    dataset = BDD100KDataset(split='train', net_type='deeplab', ignore_index=4, debug=True,affine_augmenter=affine_augmenter, image_augmenter=image_augmenter, target_size=(1280, 1280))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    palette = np.array([[0, 0, 0],      # black
                        [255, 0, 0],    # red
                        [0, 0, 255],    # blue
                        [255, 0, 255],  # pink
                        [255, 255, 0]]) # yellow

    for i, batched in enumerate(dataloader):
        images, labels = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(20, 48))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(minmax_normalize(images[j], norm_range=(0, 1), orig_range=(-1, 1)))
                axes[j][1].imshow(palette[labels[j]])
                axes[j][0].set_xticks([])
                axes[j][0].set_yticks([])
                axes[j][1].set_xticks([])
                axes[j][1].set_yticks([])
            plt.savefig('dataset/bdd100k.png')
            plt.close()
        break
