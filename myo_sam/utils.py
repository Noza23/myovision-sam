import torch
from torch.nn import functional as F

def sample_initial_points(
    gt_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random point from each mask in gt_masks. Point
        coords are in the format [x, y] and is located on the instance.
    Args:
        gt_masks (torch.Tensor): GT masks of shape (N, 1, 1024, 1024).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Initial points and labels
            of shape (N, 1, 2) and (N, 1) respectively.
    """
    assert(len(gt_masks.shape) == 4)
    initial_points = []
    for gt_mask in gt_masks:
        coords_pos = torch.nonzero(gt_mask.squeeze(0))
        rand_id = torch.randint(0, len(coords_pos), (1, ))
        initial_points.append(
            # Flip the coordinates to match the format of the model [x, y]
            torch.flip(coords_pos[rand_id].to(torch.float32), dims=(1, ))
        )
    return (torch.stack(initial_points, dim=0), torch.ones(len(gt_masks), 1))


def sample_points_from_error_region(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random point from the error region given the
        GT and predicted masks.
    Args:
        gt_masks (torch.Tensor): GT masks (N, 1, 1024, 1024)
        pred_masks (torch.Tensor): Prediction masks (N, 1, 1024, 1024)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Point coordinates and labels
            of shape (N, 1, 2) and (N, 1) respectively.
    """
    assert(gt_masks.shape == pred_masks.shape)
    N, _, _, _ = gt_masks.shape
    point_coords, labels = [], []
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        error_region = torch.logical_xor(pred_mask, gt_mask)
        error_region_coords = torch.nonzero(error_region.squeeze(0))
        if len(error_region_coords) == 0:
            # No error region, sample from the mask ramdomly
            coords_pos = torch.nonzero(gt_mask.squeeze(0))
            rand_id = torch.randint(0, len(coords_pos), (1, ))
            point_rand = coords_pos[rand_id]
        else:
            rand_id = torch.randint(0, len(error_region_coords), (1, ))
            point_rand = error_region_coords[rand_id]
        # Background: 0 or Foreground: 1 label
        label = gt_mask[:, point_rand[0, 0], point_rand[0, 1]] == 1
        point_coords.append(
            # Flip the coordinates to match the format of the model [x, y]
            torch.flip(point_rand.to(torch.float32), dims=(1, ))
        )
        labels.append(label.to(torch.float32))
    return (torch.stack(point_coords, dim=0), torch.stack(labels, dim=0))

def normalize_pixels(
    x: torch.Tensor, mean: list[float], std: list[float]
) -> torch.Tensor:
    """
    Normalizes pixel values.
    Args:
        x (torch.Tensor): Tensor to normalize.
        mean (list[float]): mean of each channel to normalize with.
        std (list[float]): std of each channel to normalize with.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    assert(len(mean) == len(std))
    mean_tensor = torch.Tensor(mean).view(-1, 1, 1)
    std_tensor = torch.Tensor(std).view(-1, 1, 1)
    return (x - mean_tensor) / std_tensor

def pad_to_square(x: torch.Tensor, size: int=1024) -> torch.Tensor:
    """
    Zero pads x to a square of size (size, size).
    Args:
        x (torch.Tensor): Tensor to pad.
        size (int): Size of square to pad to.

    Returns:
        torch.Tensor: Padded tensor.
    """
    h, w = x.shape[-2:]
    padh, padw = size - h, size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x