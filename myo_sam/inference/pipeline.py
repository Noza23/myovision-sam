from functools import cached_property
from typing import Optional, Any
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field, root_validator, field_validator

from .predictors.stardist_predictor import StarDistPredictor, StarDistConfig
from .predictors.myosam_predictor import MyoSamPredictor, MyoSamConfig

from .models.base import Myotubes, Nucleis, NucleiClusters
from .models.information import InformationMetrics
from .models.performance import PerformanceMetrics
from .models.result import MyoSamInferenceResult


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
    measure_unit: int = Field(
        description="The measure unit of the images.", default=1
    )

    @cached_property
    def myotube_image_np(self) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(self.myotube_image), cv2.COLOR_BGR2RGB)

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

    def execute(self) -> MyoSamInferenceResult:
        """Execute the pipeline."""
        if self.myotube_image:
            myosam_predictor = MyoSamPredictor(config=self.myosam_config)
            myotube_pred = myosam_predictor.predict(
                self.myotube_image_np, self.measure_unit
            )
            myotubes = Myotubes.model_validate({"myo_objects": myotube_pred})
        else:
            myotubes = Myotubes()

        if self.nuclei_image:
            stardist_pred = StarDistPredictor(config=self.stardist_config)
            nuclei_pred = stardist_pred.predict(
                self.nuclei_image_np, self.measure_unit
            )
            nucleis = Nucleis.parse_nucleis(**nuclei_pred, myotubes=myotubes)
            clusters = NucleiClusters.compute_clusters(nucleis)
        else:
            nucleis = Nucleis()
            clusters = NucleiClusters.compute_clusters(nucleis)

        info = InformationMetrics(
            myotubes=myotubes, nucleis=nucleis, nuclei_clusters=clusters
        )
        perf = PerformanceMetrics.compute_performance(
            predictions=myotubes, ground_truths=myotubes
        )

        result = MyoSamInferenceResult(
            myotube_image=self.myotube_image,
            nuclei_image=self.nuclei_image,
            information_metrics=info,
            performance_metrics=perf,
        )
        return result
