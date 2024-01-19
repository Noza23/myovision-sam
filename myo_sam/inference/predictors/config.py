from pydantic import BaseModel, Field


class AmgConfig(BaseModel):
    """The configuration of a AMG framework."""

    points_per_side: int = Field(
        description="Number of points per side", default=64, ge=1, step=5
    )
    points_per_batch: int = Field(
        description="Number of points per batch", default=64, ge=1, step=5
    )
    pred_iou_thresh: float = Field(
        description="Threshold for predicted IoU",
        default=0.8,
        ge=0,
        le=1,
        step=0.05,
    )
    stability_score_thresh: float = Field(
        description="Threshold for stability score",
        default=0.92,
        ge=0,
        le=1,
        step=0.05,
    )
    stability_score_offset: float = Field(
        description="Offset in computing stability score",
        default=1.0,
        ge=0,
        step=0.05,
    )
    box_nms_thresh: float = Field(
        description="Threshold for filtering duplicates",
        default=0.7,
        ge=0,
        le=1,
        step=0.05,
    )
    crop_n_layers: int = Field(
        description="Rerun algorithm on crops", default=1, ge=0, le=4, step=1
    )
    crop_nms_thresh: float = Field(
        description="NMS threshold for cropping",
        default=0.7,
        ge=0,
        le=1,
        step=0.05,
    )
    crop_overlap_ratio: float = Field(
        description="Overlap ratio in cropping",
        default=0.34,
        ge=0,
        le=1,
        step=0.05,
    )
    crop_n_points_downscale_factor: int = Field(
        description="Point downscale factor for cropping",
        default=2,
        ge=1,
        step=1,
    )
    min_mask_region_area: int = Field(
        description="Threshold for Minimum area of mask",
        default=100,
        ge=0,
        step=50,
    )
