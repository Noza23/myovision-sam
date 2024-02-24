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

    tp: int = Field(description="Number of true positives")
    fp: int = Field(description="Number of false positives")
    fn: int = Field(description="Number of false negatives")
    precision: float = Field(description="Precision")
    recall: float = Field(description="Recall")
    accuracy: float = Field(description="Accuracy")
    panoptic_quality: float = Field(description="Panoptic Quality")
    mean_matching_score: float = Field(
        description="Mean matching score (either iou or ior)"
    )
    normalized_matching_score: float = Field(
        description="Normalized matching score"
    )
    mean_nsd: float = Field(description="Mean nsd (only for myotubes)")
    normalized_nsd: float = Field(
        description="Normalized nsd (only for myotubes)"
    )
    mean_iou: float = Field(
        description="Mean iou (only for mask_ior matching)"
    )
    normalized_iou: float = Field(
        description="Normalized iou (only for mask_ior matching)"
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
    ) -> dict:
        """
        Calculates performance metrics of MyoSAM inference.

        Args:
            ref_loc (list of np.ndarray): List of reference binary masks.
            pred_loc (list of np.ndarray): List of predicted binary masks.
            localization (str): Localization method, either 'mask_iou' (intersection over union) or 'mask_ior' (intersection over reference).
            thresh (float): Matching threshold.
            flag_fp_in (bool): Flag for double detection.
            If True, predicted masks other than the best match that match/surpass the threshold are considered false positives.
            object_type (str): Object type, either 'myotubes' or 'nuclei'.
            im_shape (tuple): Image shape (height, width).

        Returns:
            dict: Dictionary with performance metrics.
            metrics in dict: number of true positives
                            number of false positives
                            number of false negatives
                            precision
                            recall
                            accuracy
                            panoptic quality
                            mean matching score (either iou or ior)
                            normalized matching score
                            mean nsd (only for myotubes)
                            normalized nsd (only for myotubes)
                            mean iou (only for mask_ior matching)
                            normalized iou (only for mask_ior matching)
        """
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
            "true positives": tp,
            "false positives": fp,
            "false negatives": fn,
            "precision": ppv,
            "recall": sens,
            "accuracy": ap,
            "panoptic quality": pq,
            "mean matching score": score_localization,
            "normalised matching score": norm_score_localization,
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
                {"mean nsd": mean_nsd, "normalised nsd": norm_mean_nsd}
            )
        else:
            result_dict.update({"mean nsd": None, "normalised nsd": None})

        # IOU calculation for 'mask_ior'
        if localization == "mask_ior":
            iou_lst = []
            for pred_mask, ref_mask in zip(tp_pred_loc, tp_ref_loc):
                binary_measures = BPM(pred_mask, ref_mask)
                iou_lst.append(binary_measures.intersection_over_union())

            mean_iou = np.mean(iou_lst)
            norm_mean_iou = (mean_iou * tp) / len(ref_loc)

            result_dict.update(
                {"mean iou": mean_iou, "normalised iou": norm_mean_iou}
            )

        return cls.model_validate(result_dict)
