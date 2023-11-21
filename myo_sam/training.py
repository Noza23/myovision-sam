import torch

def sample_initial_points(
    gt_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random point from each mask in gt_masks. Point
    coords are in the format [x, y] and is located on the instance.

    Args:
        gt_masks (torch.Tensor): GT masks of shape (B, 1, 1024, 1024)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Initial points and labels
            of shape (B, 1, 2) and (B, 1) respectively.
    """
    if len(gt_masks.shape) != 4:
        raise ValueError('gt_masks should have shape (B, C, H, W)')
    initial_points = []
    for gt_mask in gt_masks:
        coords_pos = torch.nonzero(gt_mask.squeeze(0))
        rand_id = torch.randint(0, len(coords_pos), (1, ))
        initial_points.append(
            # Flip the coordinates to match the format of the model [x, y]
            torch.flip(coords_pos[rand_id].to(torch.float32), dims=(1, ))
        )
    return (torch.stack(initial_points, dim=0), torch.ones(len(gt_masks), 1))


def sample_point_from_error_region(
    gt_masks: torch.Tensor, # (B, 1, 1024, 1024)
    pred_masks: torch.Tensor, # (B, 1, 1024, 1024)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a random point from the error region given the
    GT and predicted masks.
    
    Args:
        gt_masks (torch.Tensor): GT masks (B, 1, 1024, 1024)
        pred_masks (torch.Tensor): Prediction masks (B, 1, 1024, 1024)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Point coordinates and labels
            of shape (B, 1, 2) and (B, 1) respectively.
    """
    if gt_masks.shape != pred_masks.shape:
        raise ValueError('gt_masks and pred_masks should have the same shape')
    B, _, _, _ = gt_masks.shape
    point_coords, labels = [], []
    for i in range(B):
        error_region = torch.logical_xor(pred_masks[i], gt_masks[i])
        error_region_coords = torch.nonzero(error_region.squeeze(0))
        rand_id = torch.randint(0, len(error_region_coords), (1, ))
        point_rand = error_region_coords[rand_id]
        # Background: 0 or Foreground: 1 label
        label = gt_masks[i, :, point_rand[0, 0], point_rand[0, 1]] == 1
        point_coords.append(
            # Flip the coordinates to match the format of the model [x, y]
            torch.flip(point_rand.to(torch.float32), dims=(1, ))
        )
        labels.append(label.to(torch.float32))
    return (torch.stack(point_coords, dim=0), torch.stack(labels, dim=0))