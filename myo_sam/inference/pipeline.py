from functools import cached_property
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .predictors import StarDistPredictor, MyoSamPredictor

from .models.base import Myotubes, Nucleis, NucleiClusters
from .models.information import InformationMetrics
from .models.performance import PerformanceMetrics
from .models.result import MyoSamInferenceResult

from .utils import hash_array


class Pipeline(BaseModel):
    """The pipeline of a MyoSam inference."""

    validate_assignment = True
    myotube_image: Optional[str] = Field(
        description="Path to Myotube Image",
        default=None,
        validate_default=False,
    )
    nuclei_image: Optional[str] = Field(
        description="Path to Myoblast Image",
        default=None,
        validate_default=False,
    )

    measure_unit: int = Field(
        description="The measure unit of the images.", default=1
    )
    stardist_predictor: StarDistPredictor = Field(
        description="The predictor of a StarDist inference.",
        default=StarDistPredictor(),
        exclude=True,
    )

    myosam_predictor: MyoSamPredictor = Field(
        description="The predictor of a MyoSam inference.",
        default=None,
        exclude=True,
    )

    def clear_cache(self) -> None:
        """Clear cached properties."""
        del self.myotube_image_np, self.nuclei_image_np
        del self.myotube_image_hash, self.nuclei_image_hash

    @cached_property
    def myotube_image_np(self) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(self.myotube_image), cv2.COLOR_BGR2RGB)

    @cached_property
    def nuclei_image_np(self) -> np.ndarray:
        return cv2.imread(self.nuclei_image, cv2.IMREAD_GRAYSCALE)

    @cached_property
    def myotube_image_hash(self) -> str:
        return hash_array(self.myotube_image_np)

    @cached_property
    def nuclei_image_hash(self) -> str:
        return hash_array(self.nuclei_image_np)

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_exists(cls, v) -> str:
        if v:
            if not Path(v).exists():
                raise ValueError(f"File {v} does not exist.")
        return v

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_extension(cls, v) -> str:
        if v:
            if not Path(v).suffix not in [".png", ".jpeg", ".tif", ".tiff"]:
                raise ValueError(f"File {v} must be a png, jpeg or tif.")
        return v

    def execute(
        self,
        myotubes_cached: Optional[str] = None,
        nucleis_cached: Optional[str] = None,
    ) -> MyoSamInferenceResult:
        """
        Execute the pipeline of the inference.
            If myotubes or nucleis are not propvided, cached.
            They will be predicted.
        """
        if not self.myotube_image and not self.nuclei_image:
            raise ValueError(
                "At least one of myotube or nuclei image must be provided."
            )
        if self.myotube_image:
            if not myotubes_cached:
                myotube_pred = self.myosam_predictor.predict(
                    self.myotube_image_np, self.measure_unit
                )
                myotubes = Myotubes.model_validate(
                    {"myo_objects": myotube_pred}
                )
            else:
                myotubes = Myotubes.model_validate(myotubes_cached)
        else:
            myotubes = Myotubes()

        if self.nuclei_image:
            if not nucleis_cached:
                nuclei_pred = self.stardist_predictor.predict(
                    self.nuclei_image_np
                )
                nucleis = Nucleis.parse_nucleis(
                    **nuclei_pred,
                    myotubes=myotubes,
                    measure_unit=self.measure_unit,
                )
            else:
                nucleis = Nucleis.model_validate(nucleis_cached)
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
