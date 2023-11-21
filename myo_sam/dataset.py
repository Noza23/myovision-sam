import torch
from torch.utils.data import Dataset
from torchvision.io import read_image  
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import rle_to_mask
import os, json

class MyoData(Dataset):
    def __init__(self, data_dir: str):
        self.transform = ResizeLongestSide(1024)
        self.data_dir = data_dir
        self.segmentations = [
            fn for fn in os.listdir(data_dir) if fn.endswith('.json')
        ]

    def __len__(self):
        return len(self.segmentations)

    def __getitem__(self, idx: int):
        seg = json.load(
            open(os.path.join(self.data_dir, self.segmentations[idx]), 'r')
        )
        
        image = read_image(
            os.path.join(self.data_dir, seg["patch"]["file_name_patch"])
        )
        # Rezize longest side to 1024
        image = torch.from_numpy(
            self.transform.apply_image(image.permute(1, 2, 0).numpy())
        ).permute(2, 0, 1)
        transformed_masks = [
            torch.from_numpy(
                self.transform.apply_image(rle_to_mask(mk).astype("uint8"))
            )
            for mk in seg["annotations"]
        ]
        gt_instances = torch.stack(transformed_masks)
        # return (C, H, W), (N, H, W)
        return image, gt_instances
