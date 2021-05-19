import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from dataset import CatDogsDataset
from tqdm import tqdm


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return (tuple(zip(*batch)))


def get_train_transform():
    return A.Compose([
        A.Resize(224, 224),
        # A.CenterCrop(width=256, height=256, p=1),
        A.Flip(0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


if __name__ == '__main__':

    train_path = r'C:\Users\Andrey Ilyin\Desktop\fasterrcnn\train'
    valid_path = r'C:\Users\Andrey Ilyin\Desktop\fasterrcnn\valid'

    train_dataset = CatDogsDataset(train_path, get_train_transform())
    valid_dataset = CatDogsDataset(valid_path, get_valid_transform())

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=4,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=8,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_epochs = 50
    num_classes = 3
    # Create the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    # replace the pre-trained head with a new one
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None

    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_hist.send(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if itr % 50 == 0:
                print(f'Iteration {itr} loss: {loss_value}')

            itr += 1
        if lr_scheduler is not None:
            lr.scheduler.step()

        print(f'Epoch #{epoch} loss: {loss_hist.value}')
