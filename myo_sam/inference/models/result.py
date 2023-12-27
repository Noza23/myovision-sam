from pydantic import BaseModel, Field, field_validator
from .performance import PerformanceMetrics
from .information import InformationMetrics
from pathlib import Path


class MyoSamInferenceResult(BaseModel):
    """
    The result of a MyoSam inference.
    """

    myotube_image: str = Field(..., example="~/myotube_image.png")
    myoblast_image: str = Field(..., example="~/myoblast_image.png")

    # The performance metrics
    performance_metrics: PerformanceMetrics = Field(
        ..., example=PerformanceMetrics()
    )
    # The information metrics
    information_metrics: InformationMetrics = Field(
        ..., example=InformationMetrics()
    )

    @field_validator("myotube_image", "myoblast_image")
    def validate_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist.")
        return v
