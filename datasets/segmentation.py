import os
from os import path

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as F


class WaterSegmentationDataset(Dataset):
    CLASSES = ["water"]
    OUTPUT_SIZE = [224, 224]

    def __init__(self, images_dir: str, labels_dir: str, augmentations: list | None = None):
        self.images_dir: str = path.abspath(images_dir)
        self.labels_dir: str = path.abspath(labels_dir)
        self.sample_filenames: list[str] = self.__get_sample_filenames(self.images_dir)
        self.__images: list[np.ndarray] = self.__load_images_to_memory()
        self.__masks: list[np.ndarray] = self.__load_masks_to_memory()
        self.__augmentations: list | None = augmentations

    def __len__(self) -> int:
        return len(self.sample_filenames)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = self.__images[index]
        mask = self.__masks[index]
        if self.__augmentations:
            transform = A.Compose(self.__augmentations)
            transformed = transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        img = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        target = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(dtype=torch.float32)
        return img, target

    def __get_sample_filenames(self, images_dir: str) -> list[str]:
        sample_ids = []
        files = os.listdir(images_dir)
        for file in files:
            sample_ids.append(file)
        return sorted(sample_ids)

    def __open_as_rgb_image(self, img_path: str) -> Image.Image:
        with Image.open(img_path) as im:
            rgb_im = im.convert('RGB')
        return rgb_im

    def __load_images_to_memory(self) -> list[np.ndarray]:
        images = []
        for filename in self.sample_filenames:
            image_path = f"{self.images_dir}/{filename}"
            rgb_im = self.__open_as_rgb_image(image_path)
            image = F.pil_to_tensor(rgb_im).float() / 255
            image = F.resize(image, self.OUTPUT_SIZE)
            img_array = np.transpose(image.numpy(), (1, 2, 0))
            images.append(img_array)
        return images

    def __load_masks_to_memory(self) -> list[np.ndarray]:
        masks = []
        for filename in self.sample_filenames:
            label_path = f"{self.labels_dir}/{filename}"
            rgb_mask = self.__open_as_rgb_image(label_path)
            mask = F.pil_to_tensor(rgb_mask)
            mask = F.resize(mask, self.OUTPUT_SIZE)
            mask = self.__encode_water(mask).unsqueeze(0)
            mask_array = np.transpose(mask.numpy(), (1, 2, 0))
            masks.append(mask_array)
        return masks

    def __encode_water(self, mask) -> Tensor:
        greyscale_mask = mask.sum(axis=0) / 3
        hot_encoded_mask = greyscale_mask > 200
        encoded_water_mask = hot_encoded_mask.long()
        return encoded_water_mask

