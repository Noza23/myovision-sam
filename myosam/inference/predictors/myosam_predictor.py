from pathlib import Path
from typing import Any, Union
from itertools import chain

import numpy as np
import cv2


from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from myosam.inference.build_myosam import build_myosam_inference

from .config import AmgConfig
from .amg import CustomAutomaticMaskGenerator
from .utils import (
    split_image_into_patches,
    merge_masks_at_splitponits,
    is_on_edge,
)
from ..models.base import Myotube


class MyoSamPredictor:
    """The predictor of a MyoSam inference."""

    def __init__(
        self,
        amg_config: AmgConfig = AmgConfig(),
        model: Union[Sam, None] = None,
        mu: float = 1,
    ) -> None:
        self.amg_config = amg_config
        self.model = model
        self.measure_unit = mu

    def set_measure_unit(self, mu: float) -> None:
        """Set the measure unit of the images."""
        self.measure_unit = mu

    def set_model(self, checkpoint: str, device: str) -> None:
        """Set the model of the predictor."""
        # if device == "cpu":
        #     raise ValueError("Running MyoSAM on CPU is not supported.")
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
    def amg(self) -> CustomAutomaticMaskGenerator:
        if not self.model:
            raise ValueError("Model must be set.")
        return CustomAutomaticMaskGenerator(
            self.model, **self.amg_config.model_dump()
        )

    @property
    def predictor(self) -> SamPredictor:
        if not self.model:
            raise ValueError("Model must be set.")
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
        method = (
            cv2.CHAIN_APPROX_NONE if all_contours else cv2.CHAIN_APPROX_SIMPLE
        )
        patch_size = (1500, 1500)
        grid, patches = split_image_into_patches(image, patch_size)
        pred_pre = []
        for i, patch in enumerate(patches):
            print(f"> Predicting patch {i + 1}/{len(patches)}...", flush=True)
            pred_dict = self.amg.generate(patch)
            r, c = i // grid[1], i % grid[1]  # row, column
            offset = np.array([c * patch_size[1], r * patch_size[0]])[None, :]
            for i, pred in enumerate(pred_dict):
                pred_pre.append(
                    {
                        "identifier": i,
                        "roi_coords": cv2.findContours(
                            pred["segmentation"].astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            method,
                        )[0][0]
                        + offset,
                        "measure_unit": self.measure_unit,
                        "pred_iou": pred["predicted_iou"],
                        "stability": pred["stability_score"],
                        "rgb_repr": patch[pred["segmentation"]].tolist(),
                    }
                )
        return self.postprocess_pred(pred_pre, grid, patch_size, image.shape)

    def postprocess_pred(
        self,
        pred_dict: list[dict[str, Any]],
        grid: tuple[int, int],
        patch_size: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Postprocess myosam prediction results."""
        pred_post = []
        conts, ids = merge_masks_at_splitponits(
            [pred["roi_coords"] for pred in pred_dict],
            grid,
            patch_size,
            iou_threshold=0.5,
            max_offset=1,
        )

        for i, lst in enumerate(ids):
            pred_post.append(
                {
                    "identifier": i,
                    "roi_coords": conts[i].squeeze().tolist(),
                    "measure_unit": self.measure_unit,
                    "pred_iou": np.mean(
                        [pred_dict[i]["pred_iou"] for i in lst],
                        dtype=np.float16,
                    ).item(),  # new mean iou
                    "stability": np.mean(
                        [pred_dict[i]["stability"] for i in lst],
                        dtype=np.float16,
                    ).item(),  # new mean stability
                    "rgb_repr": list(
                        chain.from_iterable(
                            [pred_dict[i]["rgb_repr"] for i in lst]
                        )  # new merged rgb_repr
                    ),
                    "is_on_edge": is_on_edge(conts[i], [0, image_shape[0]], 0)
                    or is_on_edge(conts[i], [0, image_shape[1]], 1),
                }
            )
        return pred_post

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
                "roi_coords": coords.squeeze().tolist(),
                "measure_unit": self.measure_unit,
                "pred_iou": score.item(),
                "stability": None,
                "rgb_repr": image[mask].tolist(),
            }
        )
