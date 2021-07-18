from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class RoadSegmDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None, mode="train"):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            mask_name = str(image_filename[:image_filename.index("_")]+"_road"+image_filename[image_filename.index("_"):]).replace(".jpg", ".png")
            mask_img = cv2.cvtColor(cv2.imread(
                os.path.join(self.masks_directory, mask_name)), cv2.COLOR_BGR2RGB)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[mask_img[:,:,2] > 0] = 1
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)


        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


class RoadInferenceDataset(Dataset):
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, original_size

        