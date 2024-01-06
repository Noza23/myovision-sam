from pathlib import Path
from typing import Any, Union

import numpy as np
import cv2

from segment_anything import SamAutomaticMaskGenerator

from myo_sam.myosam import MyoSam
from myo_sam.build_myosam import build_myosam_inference

from .config import AmgConfig


class MyoSamPredictor:
    """The predictor of a MyoSam inference."""

    def __init__(self) -> None:
        self.amg_config: AmgConfig = AmgConfig()
        self.model: Union[MyoSam, None] = None

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
        return SamAutomaticMaskGenerator(self.model, **self.amg_config)

    def update_amg_config(self, config: dict[str, Any]) -> None:
        """Update the configuration of the predictor."""
        self.amg_config = AmgConfig.model_validate(config)

    def predict(self, image: np.ndarray, mu: float) -> list[dict[str, Any]]:
        """
        Predict the segmentation of the image.

        Args:
            image: RGB image to predict.
            mu: The measure unit of the image.
        """
        pred_dict = self.amg.generate(image)
        return self.postprocess_pred(pred_dict, image, mu)

    def postprocess_pred(
        self, pred_dict: list[dict[str, Any]], image: np.ndarray, mu: float
    ) -> list[dict[str, Any]]:
        """Postprocess myosam prediction results."""
        pred_post = []
        for i, pred in enumerate(pred_dict):
            roi_cords = cv2.findContours(
                pred["segmentation"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            pred_post.append(
                {
                    "identifier": i,
                    "roi_coords": roi_cords[0][0],
                    "measure_unit": mu,
                    "pred_iou": pred["predicted_iou"],
                    "stability": pred["stability_score"],
                    "rgb_repr": image[pred["segmentation"]].tolist(),
                }
            )
        return pred_dict
