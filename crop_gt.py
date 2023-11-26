import os

import cv2

from dataset import TrainSet

if __name__ == "__main__":
    data_path = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset"
    data = TrainSet(data_path)
    for sample in data:
        img, label, img_path = sample
        img = cv2.imread(os.path.join(data_path, img_path))
        for box in label:
            y, x, h, w = box
            cropped_img = img[
                round(y - h / 2) : round(y + h / 2), round(x - w / 2) : round(x + w / 2)
            ]
            try:
                cv2.imwrite(os.path.join(data_path, "cropped", img_path), cropped_img)
            except:
                pass
