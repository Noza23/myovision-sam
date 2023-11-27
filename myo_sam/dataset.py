import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image  
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import rle_to_mask
import os, json
from .utils import pad_to_square, normalize_pixels

class MyoData(Dataset):
    def __init__(self, data_dir: str, train: bool=True, split: float=0.8):
        self.resize_longest_side = ResizeLongestSide(1024)
        self.data_dir = data_dir
        self.segmentations = [
            fn for fn in os.listdir(data_dir) if fn.endswith('.json')
        ]
        split_index = int(split * len(self.segmentations))
        if train:
            self.segmentations = self.segmentations[: split_index]
        else:
            self.segmentations = self.segmentations[split_index: ]

    def transform_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Performs preprocessing steps on an image of shape (H, W, C):
            - Resize longest side to 1024
            - Convert to torch.Tensor and permute to (H, W, C)
            - Normalize pixel values and pad to a square input
        
        Returns:
            (torch.Tensor): Preprocessed image of shape (C, H, W).
        """
        image = self.resize_longest_side.apply_image(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image= normalize_pixels(
            image,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
        image = pad_to_square(image)
        return image

    def transform_mask(self, masks: np.ndarray) -> torch.Tensor:
        """
        Performs preprocessing steps on a mask of shape (H, W):
            - Resize longest side to 1024
            - Convert to torch.Tensor
            - Pad to a square input
        
        Returns:
            (torch.Tensor): Preprocessed mask of shape (H, W).
        """
        masks = self.resize_longest_side.apply_image(masks)
        masks = torch.from_numpy(masks)
        masks = pad_to_square(masks)
        return masks
    

    def __len__(self):
        return len(self.segmentations)

    def __getitem__(self, idx: int):
        seg = json.load(
            open(os.path.join(self.data_dir, self.segmentations[idx]), 'r')
        )
        
        image = read_image(
            os.path.join(self.data_dir, seg["patch"]["file_name_patch"])
        )
        image = self.transform_image(image.permute(1, 2, 0).numpy())

        transformed_masks = [
            self.transform_mask(rle_to_mask(mk).astype("uint8"))
            for mk in seg["annotations"]
        ]
        assert(image.shape[-2: ] == transformed_masks[0].shape)
        # return (C, 1024, 1024), (N, 1024, 1024)
        return image, torch.stack(transformed_masks)
