import os
import sys

import torch
from PIL import Image
from torchvision import transforms

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
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Resize((image_size, image_size)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )
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
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, label, img_path

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
    trainset = TrainSet("/local/omp/eagle_eyes_hackathon/eagle_eyes_hackathon_dataset")
    print(len(trainset))
    print(trainset[0][0].size)
