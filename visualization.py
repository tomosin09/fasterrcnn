import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from dataset import CatDogsDataset
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    train_path = r'C:\Users\Andrey Ilyin\Desktop\fasterrcnn\train'
    valid_path = r'C:\Users\Andrey Ilyin\Desktop\fasterrcnn\valid'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def get_train_transform():
        return A.Compose([
            A.Resize(416, 416),
            A.CenterCrop(width=256, height=256, p=1),
            A.Flip(0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    def get_valid_transform():
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    trainDataset = CatDogsDataset(train_path, get_train_transform())
    validDataset = CatDogsDataset(valid_path, get_valid_transform())
    train_data_loader = DataLoader(
        trainDataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        validDataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Check our dataset
    images, targets, image_ids = next(iter(valid_data_loader))
    # print(f'images:{images}')
    # print(f'targets:{targets}')
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    print(targets)
    name = None

    for i in range(len(images)):
        Img = images[i].permute(1, 2, 0).cpu().numpy()
        bboxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
        labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
        for box in bboxes:
            cv2.rectangle(Img,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (220, 0, 0), 3)
        plt.subplot(4, 4, i + 1);
        plt.imshow(Img)
        for lbl in labels:
            if lbl == 1:
                name = 'cat'
            else:
                name = 'dog'
        plt.title(name)
        plt.xticks([]), plt.yticks([])
    plt.show()
