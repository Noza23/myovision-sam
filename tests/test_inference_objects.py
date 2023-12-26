import numpy as np
from myo_sam.inference.models.base import Nuclei, Myotube

nuclei_pred = np.load(
    "tests/data/stardist_res.npy", allow_pickle=True
).tolist()
myotube_pred = np.load("tests/data/myosam_res.npy", allow_pickle=True).tolist()


nuclei_info = {
    "identifier": 1,
    "roi_coords": np.flip(nuclei_pred["coord"][0].astype(int).T, 1).tolist(),
    "measure_unit": 1,
    "myotube_ids": [2, 6],
    "centroid": np.flip(nuclei_pred["points"][0], 0).tolist(),
    "prob": nuclei_pred["prob"][0],
}

myotube_info = {
    "identifier": 1,
    "roi_coords": myotube_pred[0]["segmentation"].squeeze(),
    "measure_unit": 1,
    "pred_iou": myotube_pred[0]["predicted_iou"],
    "stability": myotube_pred[0]["stability_score"],
    "rgb_repr": [[1, 2, 10], [1, 6, 7], [10, 30, 20]],
}


def test_nuclei():
    assert Nuclei(**nuclei_info)


def test_myotube():
    assert Myotube(**myotube_info)
