from pydantic import BaseModel, Field


class AmgConfig(BaseModel):
    """AMG configuration"""

    points_per_side: int = Field(
        description="Number of points per side", default=64, ge=1
    )
    points_per_batch: int = Field(
        description="Number of points per batch", default=64, ge=1
    )
    pred_iou_thresh: float = Field(
        description="Threshold for predicted IoU",
        default=0.8,
        ge=0,
        le=1,
        step=0.01,
    )
    stability_score_thresh: float = Field(
        description="Threshold for stability score",
        default=0.92,
        ge=0,
        le=1,
        step=0.01,
    )
    stability_score_offset: float = Field(
        description="Offset in computing stability score",
        default=1.0,
        ge=0,
        step=0.01,
    )
    box_nms_thresh: float = Field(
        description="Threshold for filtering duplicates",
        default=0.7,
        ge=0,
        le=1,
        step=0.01,
    )
    crop_n_layers: int = Field(
        description="Rerun algorithm on crops", default=1, ge=0, le=4
    )
    crop_nms_thresh: float = Field(
        description="NMS threshold for cropping",
        default=0.7,
        ge=0,
        le=1,
        step=0.01,
    )
    crop_overlap_ratio: float = Field(
        description="Overlap ratio in cropping",
        default=0.34,
        ge=0,
        le=1,
        step=0.001,
    )
    crop_n_points_downscale_factor: int = Field(
        description="Point downscale factor for cropping", default=2, ge=1
    )
    min_mask_region_area: int = Field(
        description="Threshold for Minimum area of mask", default=100, ge=0
    )
