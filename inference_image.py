from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch
import sys
import os

model = UNet(enc_chs=(3,64, 128, 256), dec_chs=(256, 128, 64), retain_dim=True, out_sz=(256, 256))
model.load_state_dict(torch.load("saved_models/road_segmentation.pth"))

images_directory = sys.argv[1]
orig_image = cv2.imread(os.path.join(images_directory))

transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

image = cv2.imread(os.path.join(images_directory))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = tuple(image.shape[:2])
transformed = transform(image=image)
image = transformed["image"]

output = model(image[None,:])
probabilities = torch.sigmoid(output.squeeze(1))
predicted_masks = (probabilities >= 0.5).float() * 1
predicted_masks = predicted_masks.cpu().numpy()

full_sized_mask = cv2.resize(predicted_masks[0], (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)


mask = np.concatenate([51*full_sized_mask[..., np.newaxis], 255*full_sized_mask[..., np.newaxis], 51*full_sized_mask[..., np.newaxis]], axis=2)

img = (0.4*mask + 0.9*orig_image).astype(np.uint8)




cv2.imwrite("road_segm_image.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))