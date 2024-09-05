from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from pycocotools.coco import COCO

SEG_COLORMAP = [
    [0, 0, 0],  # Class 0: Background (Black)
    [0, 0, 180],  # Class 1: Blue - Cup
    [0, 180, 0],  # Class 2: Green - Laptop
    [180, 0, 0]  # Class 3: Red - Person
]


class SegDataset(Dataset):
    def __init__(self, root="data/Semantic Segmentation.v3i.coco-segmentation", image_set="train", transform=None):

        if image_set not in ["train", "valid"]:
            raise ValueError(f"Invalid image_set: '{image_set}'. Must be 'train' or 'valid'.")

        # Set image folder based on the dataset type (train/valid)
        root = os.path.join(root, image_set)

        # Initialize COCO API and get image/annotation info
        self.coco = COCO(os.path.join(root, "_annotations.coco.json"))
        self.img_ids = self.coco.getImgIds()
        self.root = root
        self.transform = transform

    def _getImgInfo(self, idx):
        # Retrieve image info (file name, height, width)
        img_info = self.coco.loadImgs(ids=idx)[0]
        return img_info["file_name"], img_info['height'], img_info['width']

    def _get_mask(self, height, width, img_idx):
        # Initialize an empty RGB mask
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Get annotations for the current image
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_idx))

        # Draw each annotation's mask in the corresponding color
        for annotation in annotations:
            category_id = annotation['category_id']
            color = SEG_COLORMAP[category_id]

            # Generate the binary mask from the annotation
            mask = self.coco.annToMask(annotation)

            # Apply color to the mask
            rgb_mask[mask == 1] = color
        return rgb_mask

    def _convert_to_segmentation_mask(self, mask):
        # Convert the RGB mask to a multi-channel segmentation mask (one channel per class)
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(SEG_COLORMAP)))

        # Create a binary mask for each class
        for label_index, label in enumerate(SEG_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.img_ids)

    def __getitem__(self, item):
        # Load image and mask for the given index
        img_idx = self.img_ids[item]
        img_name, height, width = self._getImgInfo(img_idx)
        img_path = os.path.join(self.root, img_name)

        # Load the image and convert from BGR to RGB
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Generate the mask for the image
        mask = self._get_mask(height, width, img_idx)

        # Convert the RGB mask to the multi-channel segmentation mask
        mask = self._convert_to_segmentation_mask(mask)

        # Apply data augmentation or transformations, if specified
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask'].argmax(dim=2).squeeze()

        return img, mask