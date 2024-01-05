from typing import Union

from stardist.models import StarDist2D
from csbdeep.utils import normalize

import numpy as np


class StarDistPredictor:
    """The predictor of a StarDist inference."""

    def __init__(self):
        self.model: Union[StarDist2D, None] = None

    def set_model(self, checkpoint_name: str) -> None:
        """Set the model of the predictor."""
        self.model = StarDist2D.from_pretrained(checkpoint_name)

    def predict(self, image: np.ndarray, mu: float) -> np.ndarray:
        """
        Predict the segmentation of the image.

        Args:
            image: Grayscale image to predict.
            mu: The measure unit of the image.
        """
        if not self.model:
            raise ValueError("Model must be set.")
        _, pred_dict = self.model.predict_instances(self.preprocess(image))
        return self.postprocess_pred(pred_dict, mu)

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess the image for StarDist."""
        return normalize(image)

    @staticmethod
    def postprocess_pred(pred_dict: dict, mu: float) -> dict[str, np.ndarray]:
        """Postprocess stardist prediction results."""
        pred_dict.update(
            {
                "coord": np.flip(
                    pred_dict["coord"].astype(np.int32).transpose(0, 2, 1),
                    axis=2,
                ),
                "points": np.flip(pred_dict["points"].astype(np.int16), 1),
                "measure_unit": mu,
            }
        )
        return pred_dict
