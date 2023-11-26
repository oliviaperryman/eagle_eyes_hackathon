import os
import sys

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as F

from hackathon.data_utils.data_loading import (AnnotatedImageDataLoader,
                                               get_default_dataset_folder)

sys.path.append("/local/omp/eagle_eyes_hackathon/")

os.environ[
    "DEFAULT_DATASET_FOLDER"
] = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset"


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size=None):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size

        self.imgs = []
        self.img_paths = []
        self.labels = []
        self.load_dataset()

    def __getitem__(self, index):
        img, label, img_path = (
            self.imgs[index],
            self.labels[index],
            self.img_paths[index],
        )
        img = Image.open(os.path.join(self.data_root, img)).convert("RGB")
        has_anomaly = False

        # Take random crop of image
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.image_size, self.image_size)
        )
        img = F.crop(img, i, j, h, w)

        new_label = []
        for box in label:
            y, x, box_h, box_w = box
            x = x - box_w / 2
            y = y - box_h / 2
            x -= j
            y -= i

            # Check if the bounding box is within the crop
            if x > 0 and y > 0 and x + box_w < w and y + box_h < h:
                new_label.append([y, x, box_h, box_w])
                has_anomaly = True

        label = new_label

        return img, label, img_path, has_anomaly

    def __len__(self):
        return len(self.imgs)

    def load_dataset(self):
        dataloader = AnnotatedImageDataLoader.from_folder(get_default_dataset_folder())
        for case_name, case in dataloader.case_dict.items():
            for img in case.images:
                self.imgs.append(img.source_path)
                self.labels.append(
                    [annotation.ijhw_box for annotation in img.annotations]
                )
                self.img_paths.append(img.source_path)


if __name__ == "__main__":
    trainset = TrainSet(
        "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset",
        image_size=256,
    )
    print(len(trainset))
    img, label, img_path, has_anomaly = trainset[0]

    root = "data/eagle_eyes_hackathon_dataset/patches"

    for sample in trainset:
        img, label, img_path, has_anomaly = sample
        if has_anomaly:
            # draw rectangle on image
            # img1 = ImageDraw.Draw(img)
            y, x, h, w = label[0]
            y = round(y)
            x = round(x)
            h = round(h)
            w = round(w)
            # img1.rectangle(((x,y), (x+w,y+h)), fill=None, outline="red")
            img.save(os.path.join(root, "positive", img_path.split("/")[-1]))

            # create mask from bounding box
            mask = Image.new("L", (256, 256), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(((x, y), (x + w, y + h)), fill=255, outline=None)
            mask.save(os.path.join(root, "mask", img_path.split("/")[-1]))

        else:
            img.save(os.path.join(root, "negative", img_path.split("/")[-1]))
