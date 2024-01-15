from functools import cached_property
from typing import Optional, Union
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator, PrivateAttr

from .predictors.myosam_predictor import MyoSamPredictor
from .predictors.stardist_predictor import StarDistPredictor

from .models.base import Myotubes, Nucleis, NucleiClusters
from .models.information import InformationMetrics
from .models.result import MyoSamInferenceResult

from .utils import hash_bytes


class Pipeline(BaseModel):
    """The pipeline of a MyoSam inference."""

    class Config:
        validate_assignment = True

    myotube_image: Optional[Union[str, bytes]] = Field(
        description="Path to Myotube Image",
        default=None,
        validate_default=False,
    )
    myotube_image_name: Optional[str] = Field(
        description="Name of Myotube Image",
        default=None,
        validate_default=False,
    )

    nuclei_image: Optional[Union[str, bytes]] = Field(
        description="Path to Nuclei Image",
        default=None,
        validate_default=False,
    )

    nuclei_image_name: Optional[str] = Field(
        description="Name of Nuclei Image",
        default=None,
        validate_default=False,
    )

    measure_unit: float = Field(
        description="The measure unit of the images.", default=1
    )
    _stardist_predictor: StarDistPredictor = PrivateAttr(
        default=StarDistPredictor()
    )

    _myosam_predictor: MyoSamPredictor = PrivateAttr(default=MyoSamPredictor())

    def clear_cache(self) -> None:
        """Clear cached properties."""
        del self.myotube_image_np, self.nuclei_image_np

    @cached_property
    def myotube_image_np(self) -> np.ndarray:
        if not self.myotube_image:
            raise ValueError("Myotube image must be set.")
        if isinstance(self.myotube_image, bytes):
            img = cv2.imdecode(
                np.frombuffer(self.myotube_image, np.uint8), cv2.IMREAD_COLOR
            )
        elif isinstance(self.myotube_image, str):
            img = cv2.imread(self.myotube_image, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @cached_property
    def nuclei_image_np(self) -> np.ndarray:
        if not self.nuclei_image:
            raise ValueError("Nuclei image must be set.")
        if isinstance(self.nuclei_image, bytes):
            img = cv2.imdecode(
                np.frombuffer(self.nuclei_image, np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
        elif isinstance(self.nuclei_image, str):
            img = cv2.imread(self.nuclei_image, cv2.IMREAD_GRAYSCALE)
        return img

    def myotube_hash(self) -> str:
        return hash_bytes(
            (
                self.myotube_image_np.tobytes()
                + self._myosam_predictor.amg_config.model_dump_json().encode()
            )
        )

    def save_myotube_image(self, path: str) -> None:
        """Save the myotube image."""
        cv2.imwrite(
            path, cv2.cvtColor(self.myotube_image_np, cv2.COLOR_RGB2BGR)
        )

    def nuclei_hash(self) -> str:
        return hash_bytes(self.nuclei_image_np.tobytes())

    def set_nuclei_image(self, image: Union[str, bytes], name: str) -> None:
        """Set the nuclei image."""
        self.nuclei_image_name = name
        self.nuclei_image = image

    def set_myotube_image(self, image: Union[str, bytes], name: str) -> None:
        """Set the myotube image."""
        self.myotube_image_name = name
        self.myotube_image = image

    def set_measure_unit(self, mu: float) -> None:
        """Update the measure unit of the pipeline."""
        self.measure_unit = mu

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_exists(cls, v) -> str:
        if v and isinstance(v, str):
            if not Path(v).exists():
                raise ValueError(f"File {v} does not exist.")
        return v

    @field_validator("myotube_image", "nuclei_image")
    def validate_file_extension(cls, v) -> str:
        if v and isinstance(v, str):
            if not Path(v).suffix not in [".png", ".jpeg", ".tif", ".tiff"]:
                raise ValueError(f"File {v} must be a png, jpeg or tif.")
        return v

    def draw_contours_on_myotube_image(
        self,
        myotubes: Optional[Myotubes] = None,
        nucleis: Optional[Nucleis] = None,
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw the contours of the myotubes on the myotube image."""
        if not myotubes:
            myotubes = Myotubes()
        if not nucleis:
            nucleis = Nucleis()
        if not self.myotube_image:
            h, w = self.nuclei_image_np.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = self.myotube_image_np.copy()

        myo_rois = [myo.roi_coords_np for myo in myotubes]
        nuclei_rois = [nuclei.roi_coords_np for nuclei in nucleis]

        if myo_rois:
            cv2.drawContours(img, myo_rois, -1, (0, 255, 0), thickness)
        if nuclei_rois:
            cv2.drawContours(img, nuclei_rois, -1, (255, 0, 0), thickness - 1)

        return img

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
                self._myosam_predictor.set_measure_unit(self.measure_unit)
                myotube_pred = self._myosam_predictor.predict(
                    self.myotube_image_np
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
                nuclei_pred = self._stardist_predictor.predict(
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
        result = MyoSamInferenceResult(
            myotube_image=self.myotube_image_name,
            nuclei_image=self.nuclei_image_name,
            information_metrics=info,
        )
        return result
