from functools import cached_property

from pydantic import BaseModel, Field
from stardist.models import StarDist2D
from csbdeep.utils import normalize

import numpy as np


class StarDistPredictor(BaseModel):
    model_name: str = Field(
        description="The name of the StarDist model to use.",
        default="2D_versatile_fluo",
    )

    @cached_property
    def model(self) -> StarDist2D:
        return StarDist2D.from_pretrained(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the segmentation of the image."""
        _, pred_dict = self.model.predict_instances(self.preprocess(image))

        return self.postprocess_pred(pred_dict)

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess the image for StarDist."""
        return normalize(image)

    @staticmethod
    def postprocess_pred(pred_dict: dict) -> dict[str, np.ndarray]:
        """Postprocess stardist prediction results."""
        pred_dict.update(
            {
                "coord": np.flip(
                    pred_dict["coord"].astype(np.int32).transpose(0, 2, 1),
                    axis=2,
                ),
                "points": np.flip(pred_dict["points"].astype(np.int16), 1),
            }
        )
        return pred_dict
