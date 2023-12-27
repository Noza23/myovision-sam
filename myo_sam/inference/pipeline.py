# Flip stardist coordinats
# np.flip(myoblast_rois.transpose(0, 2, 1), axis=2).astype(np.int32)
from functools import cached_property
from typing import Optional, Any
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field, root_validator, field_validator

from .predictors.stardist_predictor import StarDistPredictor, StarDistConfig
from .predictors.myosam_predictor import MyoSamPredictor, MyoSamConfig

# from .models.base import Myotube, Nuclei, NucleiClusters
# from .models.information import InformationMetrics
# from .models.performance import PerformanceMetrics
# from .models.result import MyoSamInferenceResult


class Pipeline(BaseModel):
    """The pipeline of a MyoSam inference."""

    myotube_image: Optional[str] = Field(
        description="Path to Myotube Image", default=None
    )
    nuclei_image: Optional[str] = Field(
        description="Path to Myoblast Image", default=None
    )
    stardist_config: Optional[StarDistConfig] = Field(
        description="The configuration of the StarDist model.",
        default=StarDistConfig(),
    )
    myosam_config: Optional[MyoSamConfig] = Field(
        description="The configuration of the MyoSam model.",
        default=MyoSamConfig(),
    )

    @cached_property
    def myotube_image_np(self) -> np.ndarray:
        return cv2.imread(self.myotube_image)

    @cached_property
    def nuclei_image_np(self) -> np.ndarray:
        return cv2.imread(self.nuclei_image, cv2.IMREAD_GRAYSCALE)

    @root_validator(pre=True)
    def validate_images(cls, values) -> dict[str, Any]:
        if values["myotube_image"] is None and values["nuclei_image"] is None:
            raise ValueError(
                "At least one of myotube or nuclei image must be provided."
            )
        return values

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_exists(cls, v) -> str:
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist.")
        return v

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_extension(cls, v) -> str:
        if not Path(v).suffix not in [".png", ".jpeg", ".tif", ".tiff"]:
            raise ValueError(f"File {v} must have a png, jpeg or tif format.")
        return v

    def execute(self):
        stardist_pred = StarDistPredictor(config=self.stardist_config)
        myosam_predictor = MyoSamPredictor(config=self.myosam_config)
        nuclei_pred = stardist_pred.predict(self.nuclei_image_np)
        myotube_pred = myosam_predictor.predict(self.myotube_image_np)

        # add rgb repr to myosoam_predictor dict

        return nuclei_pred, myotube_pred
