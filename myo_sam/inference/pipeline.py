# Flip stardist coordinats
# np.flip(myoblast_rois.transpose(0, 2, 1), axis=2).astype(np.int32)
from functools import cached_property

import cv2
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field, root_validator, field_validator


class Pipeline(BaseModel):
    """The pipeline of a MyoSam inference."""

    myotube_image: Optional[str] = Field(
        description="Path to Myotube Image", default=None
    )
    nuclei_image: Optional[str] = Field(
        description="Path to Myoblast Image", default=None
    )

    @cached_property
    def myotube_image_np(self):
        return cv2.imread(self.myotube_image)

    @cached_property
    def nuclei_image_np(self):
        return cv2.imread(self.nuclei_image, cv2.IMREAD_GRAYSCALE)

    @root_validator(pre=True)
    def validate_images(cls, values):
        if values["myotube_image"] is None and values["nuclei_image"] is None:
            raise ValueError(
                "At least one of myotube or nuclei image must be provided."
            )
        return values

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist.")
        return v

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_extension(cls, v):
        if not Path(v).suffix not in [".png", ".jpeg", ".tif", ".tiff"]:
            raise ValueError(f"File {v} must have a png, jpeg or tif format.")
        return v

    def execute(self):
        raise NotImplementedError
