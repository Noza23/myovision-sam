# This is a slightly modified version of the original AMG class from the
# segment-anything repository. licensed under the Apache License, Version 2.0
# https://github.com/facebookresearch/segment-anything/blob/main/LICENSE


from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam


import numpy as np
import torch


from segment_anything.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    uncrop_masks,
)


class CustomAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(self, model: Sam, **kwargs):
        super().__init__(model, **kwargs)

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: tuple[int, ...],
        crop_box: list[int],
        orig_size: tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(
            points, im_size
        )
        in_points = torch.as_tensor(
            transformed_points, device=self.predictor.device
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,  # modification
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"],
            self.predictor.model.mask_threshold,
            self.stability_score_offset,
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data
