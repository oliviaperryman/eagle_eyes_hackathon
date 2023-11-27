# https://github.com/idealo/image-super-resolution
# Outdated ? py36 only

import os

import numpy as np
from ISR.models import RDN
from PIL import Image


if __name__ == "__main__":
    data_path = (
        "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/cropped/"
    )
    files = os.listdir(os.path.join(data_path, "images"))
    rdn = RDN(weights="psnr-small")

    for file in files[:10]:
        img = Image.open(os.path.join(data_path, file))
        lr_img = np.array(img)
        sr_img = rdn.predict(lr_img)

        img = Image.fromarray(sr_img)
        try:
            img.save(os.path.join(os.path.join(data_path, "superres"), file))
        except:
            pass
