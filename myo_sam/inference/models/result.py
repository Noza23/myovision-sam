from pydantic import BaseModel, Field
from .performance import PerformanceMetrics
from .information import InformationMetrics

import json


class MyoSamInferenceResult(BaseModel):
    """
    The result of a MyoSam inference.
    """

    myotube_image: str = Field(description="Myotube Image name")
    nuclei_image: str = Field(description="Myoblast Image name")

    # The information metrics
    information_metrics: InformationMetrics = Field(
        description="The information metrics of the inference.",
    )
    # The performance metrics
    performance_metrics: PerformanceMetrics = Field(
        description="The performance metrics of the inference.",
    )

    def save(self, path: str) -> None:
        """
        Save the result of the inference.

        Args:
            path: The path to save the result.
        """
        json.dump(self.model_dump(mode="json"), open(path, "w"))
