from pydantic import BaseModel, Field


class AmgConfig(BaseModel):
    """The configuration of a AMG framework."""

    points_per_side: int = Field(
        description="Number of points per side to sample.", default=64
    )
    points_per_batch: int = Field(
        description="Number of points to predict per batch", default=64
    )
    pred_iou_thresh: float = Field(
        description="Threshold for predicted IoU", default=0.8
    )
    stability_score_thresh: float = Field(
        description="Threshold for stability score", default=0.92
    )
    stability_score_offset: float = Field(
        description="Offset for stability score", default=1.0
    )
    box_nms_thresh: float = Field(description="NMS threshold", default=0.7)
    crop_n_layers: int = Field(
        description="Number of layers to crop", default=1
    )
    crop_nms_thresh: float = Field(
        description="NMS threshold for cropping", default=0.7
    )
    crop_overlap_ratio: float = Field(
        description="Overlap ratio for cropping", default=512 / 1500
    )
    crop_n_points_downscale_factor: int = Field(
        description="Downscale factor for cropping", default=2
    )
    min_mask_region_area: int = Field(
        description="Minimum area of mask region", default=100
    )
    output_mode: str = Field(description="Output mode", default="binary_mask")
