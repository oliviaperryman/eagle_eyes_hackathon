import cv2
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# img_path = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/images/SUNSHINE_COAST_GREEN_MOVER-0.png"
img_path = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/images/BREW_HUT_SNOW_SHADOWS_PERSON_WALKING-1.png"
image = Image.open(img_path)
texts = [["a person"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

image_out = cv2.imread(img_path)

score_threshold = max(scores)
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    x, y, w, h = box
    if score >= score_threshold:
        print(
            f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
        )
        cv2.rectangle(
            image_out,
            (round(x), round(y)),
            (round(x + w), round(y + h)),
            (0, 255, 0),
            2,
        )

y, x, h, w = 1246, 1870, 118, 82
cv2.rectangle(
    image_out,
    (round(x - w / 2), round(y - h / 2)),
    (round(x + w / 2), round(y + h / 2)),
    (0, 0, 255),
    2,
)

cv2.imwrite(f"detected_{label}_{score}.png", image_out)
