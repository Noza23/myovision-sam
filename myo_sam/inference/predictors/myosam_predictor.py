from pathlib import Path
from typing import Any, Union

import numpy as np
import cv2


from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from myo_sam.inference.build_myosam import build_myosam_inference

from .config import AmgConfig
from .amg import CustomAutomaticMaskGenerator
from ..models.base import Myotube


class MyoSamPredictor:
    """The predictor of a MyoSam inference."""

    def __init__(self) -> None:
        self.amg_config: AmgConfig = AmgConfig()
        self.model: Union[Sam, None] = None
        self.measure_unit: float = 1

    def set_measure_unit(self, mu: float) -> None:
        """Set the measure unit of the images."""
        self.measure_unit = mu

    def set_model(self, checkpoint: str, device: str) -> None:
        """Set the model of the predictor."""
        if device == "cpu":
            raise ValueError("Running MyoSAM on CPU is not supported.")
        if not Path(checkpoint).exists():
            raise ValueError(
                f"Checkpoint could be found: {checkpoint} does not exist."
            )
        self.model = build_myosam_inference(checkpoint).to(device)

    @property
    def name(self) -> str:
        """Return the name of the predictor"""
        return "MyoSam"

    @property
    def amg(self):
        return CustomAutomaticMaskGenerator(self.model, **self.amg_config)

    @property
    def predictor(self):
        return SamPredictor(self.model)

    def update_amg_config(self, config: dict[str, Any]) -> None:
        """Update the configuration of the predictor."""
        self.amg_config = AmgConfig.model_validate(config)

    def predict(
        self, image: np.ndarray, all_contours: bool = False
    ) -> list[dict[str, Any]]:
        """
        Predict the segmentation of the image.

        Args:
            image: RGB image to predict.
            all_contours: Whether to predict all contours or minimum required.
            see: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        """
        pred_dict = self.amg.generate(image)
        return self.postprocess_pred(pred_dict, image, all_contours)

    def postprocess_pred(
        self,
        pred_dict: list[dict[str, Any]],
        image: np.ndarray,
        all_contours: bool = False,
    ) -> list[dict[str, Any]]:
        """Postprocess myosam prediction results."""
        pred_post = []
        method = (
            cv2.CHAIN_APPROX_NONE if all_contours else cv2.CHAIN_APPROX_SIMPLE
        )

        for i, pred in enumerate(pred_dict):
            roi_cords = cv2.findContours(
                pred["segmentation"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                method,
            )
            pred_post.append(
                {
                    "identifier": i,
                    "roi_coords": roi_cords[0][0],
                    "measure_unit": self.measure_unit,
                    "pred_iou": pred["predicted_iou"],
                    "stability": pred["stability_score"],
                    "rgb_repr": image[pred["segmentation"]].tolist(),
                }
            )
        return pred_dict

    def predict_point(
        self,
        image: Union[np.ndarray, bytes],
        point: list[list[int]],
        all_contours: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict the segmentation for a single point [[x, y]]."""
        if isinstance(image, bytes):
            image = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
        self.predictor.set_image(image)
        mask, score, _ = self.predictor.predict(
            point_coords=np.array(point),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        method = (
            cv2.CHAIN_APPROX_NONE if all_contours else cv2.CHAIN_APPROX_SIMPLE
        )
        coords = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, method
        )[0][0]

        return Myotube.model_validate(
            {
                "identifier": 0,
                "roi_coords": coords,
                "measure_unit": self.measure_unit,
                "pred_iou": score.item(),
                "stability": None,
                "rgb_repr": image[mask].tolist(),
            }
        )
