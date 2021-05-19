import torch
import cv2
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt


class CatDogsDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Image folder
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'imgs'))))
        # Annotate files
        self.anns = list(sorted(os.listdir(os.path.join(root, 'txt'))))

    def __getitem__(self, index):
        # load images
        img_path = os.path.join(self.root, 'imgs', self.imgs[index])
        ann_path = os.path.join(self.root, 'txt', self.anns[index])
        img = np.array(Image.open(img_path), dtype=np.float32)
        img /= 255.0
        file = open(os.path.join(self.root, 'txt', self.anns[index]), 'r')
        ann = file.readlines()
        file.close()
        labels = []
        boxes = []
        for i in ann:
            i = i.split(' ')
            labels.append(i[:1])
            boxes.append(i[1:])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(f'y_max={boxes[:, 3]} y_min = {boxes[:, 1]} x_max={boxes[:, 2]} x_min = {boxes[:, 0]}')
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(ann_path)), dtype=torch.int64)
        # print(f'boxes: {boxes}')

        # print(f'file_name: {ann_path}')
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels[0]
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        # # target['iscrowd'] = iscrowd
        # print(f'labels: {target["labels"]}')
        if self.transforms:
            sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))) \
                .permute(1, 0)

        return image, target, ann_path

    def __len__(self):
        return len(self.imgs)



