from pydantic import BaseModel, Field
import numpy as np
from MetricsReloaded.metrics.pairwise_measures import (
    BinaryPairwiseMeasures as BPM,
)
from MetricsReloaded.utility.assignment_localization import (
    AssignmentMapping as AM,
)

from .utils import mask2cont


class PerformanceMetrics(BaseModel):
    """The performance metrics of a MyoSam inference."""

    tp: int = Field(description="Number of true positives", ge=0)
    fp: int = Field(description="Number of false positives", ge=0)
    fn: int = Field(description="Number of false negatives", ge=0)
    precision: float = Field(description="Precision", ge=0, le=1)
    recall: float = Field(description="Recall", ge=0, le=1)
    accuracy: float = Field(description="Accuracy", ge=0, le=1)
    panoptic_quality: float = Field(description="Panoptic Quality", ge=0, le=1)
    mean_matching_score: float = Field(
        description="Mean matching score (either iou or ior)", ge=0, le=1
    )
    normalized_matching_score: float = Field(
        description="Normalized matching score", ge=0, le=1
    )
    mean_nsd: float = Field(description="Mean nsd (only for myotubes)", ge=0)
    normalized_nsd: float = Field(
        description="Normalized nsd (only for myotubes)", ge=0
    )
    mean_iou: float = Field(
        description="Mean iou (only for mask_ior matching)", ge=0, le=1
    )
    normalized_iou: float = Field(
        description="Normalized iou (only for mask_ior matching)", ge=0, le=1
    )

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{k}: {v if not isinstance(v, float) else round(v, 2)}"
                for k, v in self.model_dump().items()
            ]
        )

    @classmethod
    def performance_myovision(
        cls,
        ref_loc: list[np.ndarray],
        pred_loc: list[np.ndarray],
        localization: str,
        thresh: float,
        flag_fp_in: bool,
        object_type: str,
        im_shape: tuple[int, int],
    ) -> "PerformanceMetrics":
        """
        Calculates performance metrics of MyoSAM inference.

        Args:
            ref_loc (list of np.ndarray): List of reference binary masks.
            pred_loc (list of np.ndarray): List of predicted binary masks.
            localization (str): Localization method, either 'mask_iou'
                (intersection over union) or 'mask_ior' (intersection over reference).
            thresh (float): Matching threshold.
            flag_fp_in (bool): Flag for double detection. If True, predicted
                masks other than the best match that match/surpass the threshold are considered false positives.
            object_type (str): Object type, either 'myotubes' or 'nuclei'.
            im_shape (tuple): Image shape (height, width).

        Returns:
            PerformanceMetrics: The performance metrics of the MyoSAM inference.
        """
        assert localization in ["mask_iou", "mask_ior"]
        assert object_type in ["myotubes", "nuclei"]

        # Initialize AssignmentMapping
        assignment_mapping = AM(
            pred_loc=pred_loc,
            ref_loc=ref_loc,
            localization=localization,
            thresh=thresh,
            assignment="greedy_matching",
            flag_fp_in=flag_fp_in,
            pred_prob=None,
        )

        # Resolve ambiguities and calculate metrics
        matching = assignment_mapping.resolve_ambiguities_matching()
        df_matching = matching[0]

        tp = len(
            df_matching[
                (df_matching["pred"] != -1) & (df_matching["ref"] != -1)
            ]
        )
        fp = len(df_matching[df_matching["ref"] == -1])
        fn = len(df_matching[df_matching["pred"] == -1])

        ppv = tp / (tp + fp)
        sens = tp / (tp + fn)

        score_localization = df_matching["performance"][
            (df_matching["pred"] != -1) & (df_matching["ref"] != -1)
        ].mean(axis=0)
        norm_score_localization = (score_localization * tp) / len(ref_loc)

        total_performance = df_matching["performance"][
            (df_matching["pred"] != -1) & (df_matching["ref"] != -1)
        ].sum()
        pq = total_performance / (tp + 0.5 * (fp + fn))

        ap = tp / (tp + fp + fn)

        result_dict = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ppv,
            "recall": sens,
            "accuracy": ap,
            "panoptic_quality": pq,
            "mean_matching_score": score_localization,
            "normalized_matching_score": norm_score_localization,
        }

        tp_pred_loc_idx_list = df_matching["pred"][
            (df_matching["pred"] != -1) & (df_matching["ref"] != -1)
        ].tolist()
        tp_ref_loc_idx_list = df_matching["ref"][
            (df_matching["pred"] != -1) & (df_matching["ref"] != -1)
        ].tolist()

        tp_pred_loc = [pred_loc[i] for i in tp_pred_loc_idx_list]
        tp_ref_loc = [ref_loc[i] for i in tp_ref_loc_idx_list]

        tp_pred_conts = [mask2cont(mask, (im_shape)) for mask in tp_pred_loc]
        tp_ref_conts = [mask2cont(mask, (im_shape)) for mask in tp_ref_loc]

        # NSD calculation for 'myotubes'
        if object_type == "myotubes":
            nsd_lst = []
            for pred_cont, ref_cont in zip(tp_pred_conts, tp_ref_conts):
                binary_measures = BPM(
                    pred_cont, ref_cont, dict_args={"nsd": 3}
                )
                nsd_lst.append(binary_measures.normalised_surface_distance())

            mean_nsd = np.mean(nsd_lst)
            norm_mean_nsd = (mean_nsd * tp) / len(ref_loc)

            result_dict.update(
                {
                    "mean_nsd": mean_nsd,
                    "normalized_nsd": norm_mean_nsd,
                    "mean_iou": None,
                    "normalized_iou": None,
                }
            )
        else:
            result_dict.update({"mean_nsd": None, "normalized_nsd": None})

        # IOU calculation for 'mask_ior'
        if localization == "mask_ior":
            iou_lst = []
            for pred_mask, ref_mask in zip(tp_pred_loc, tp_ref_loc):
                binary_measures = BPM(pred_mask, ref_mask)
                iou_lst.append(binary_measures.intersection_over_union())

            mean_iou = np.mean(iou_lst)
            norm_mean_iou = (mean_iou * tp) / len(ref_loc)

            result_dict.update(
                {"mean_iou": mean_iou, "normalized_iou": norm_mean_iou}
            )
        else:
            result_dict.update({"mean_iou": None, "normalized_iou": None})
        return cls.model_validate(result_dict)
