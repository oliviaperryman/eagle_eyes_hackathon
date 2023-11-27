# https://pypi.org/project/super-image/
import os

from PIL import Image
from super_image import EdsrModel, ImageLoader

if __name__ == "__main__":
    data_path = (
        "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/cropped/"
    )
    files = os.listdir(os.path.join(data_path, "images"))

    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

    for file in files:
        image = Image.open(os.path.join(data_path, "images", file))
        inputs = ImageLoader.load_image(image)
        preds = model(inputs)

        ImageLoader.save_image(
            preds, os.path.join(os.path.join(data_path, "superres"), file)
        )
        # ImageLoader.save_compare(inputs, preds, os.path.join(os.path.join(data_path,"superres"), file))
