from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
import numpy as np
import cv2

from segment_anything import SamAutomaticMaskGenerator

from myo_sam.myosam import MyoSam
from myo_sam.build_myosam import build_myosam_inference


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


class MyoSamConfig(AmgConfig):
    """The configuration of a MyoSam model."""

    checkpoint: str = Field(description="Path to model checkpoint")
    patch_size: int = Field(
        description="Patching image before processing", default=1500
    )
    device: str = Field(description="Device to run model on", default="cpu")

    @property
    def amg_config(self) -> dict:
        return AmgConfig(**self.model_dump()).model_dump()

    @field_validator("model")
    def validate_file_exists(cls, v) -> str:
        if not Path(v).exists():
            raise ValueError(f"Checkpoint could be found: {v} does not exist.")
        return v


class MyoSamPredictor(BaseModel):
    """The predictor of a MyoSam inference."""

    config: MyoSamConfig = Field(
        description="The configuration of the MyoSam model.",
        default=MyoSamConfig(),
    )

    @cached_property
    def model(self) -> MyoSam:
        model = build_myosam_inference(self.config.checkpoint)
        return model.to(self.config.device)

    @cached_property
    def amg(self):
        return SamAutomaticMaskGenerator(self.model, **self.config.amg_config)

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Predict the segmentation of the image."""
        return self.postprocess_pred(self.amg.generate(image))

    def postprocess_pred(
        self, pred_dict: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Postprocess myosam prediction results."""
        for pred in pred_dict:
            pred.update(
                {
                    "segmentation": cv2.findContours(
                        pred["segmentation"].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE,
                    )[0][0]
                }
            )
        return pred_dict
