import numpy as np

from myosam.inference.models.base import (
    Nuclei,
    Myotube,
    Nucleis,
    Myotubes,
    NucleiClusters,
)
from myosam.inference.models.information import InformationMetrics
from myosam.inference.models.result import MyoSamInferenceResult


nuclei_pred = np.load(
    "tests/data/stardist_res.npy", allow_pickle=True
).tolist()
myotube_pred = np.load("tests/data/myosam_res.npy", allow_pickle=True).tolist()


nuclei_info = {
    "identifier": 1,
    "roi_coords": np.flip(
        nuclei_pred["coord"][0].astype(np.int32).T, 1
    ).tolist(),
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


def get_myotubes():
    return Myotubes(
        myo_objects=[
            Myotube(
                identifier=i,
                roi_coords=myotube["segmentation"].squeeze(),
                measure_unit=1,
                pred_iou=myotube["predicted_iou"],
                stability=myotube["stability_score"],
                rgb_repr=[[1, 2, 10], [1, 6, 7], [10, 30, 20]],
            )
            for i, myotube in enumerate(myotube_pred)
        ]
    )


def get_nucleis(empty_myotubes=False):
    if empty_myotubes:
        myotubes = Myotubes()
    else:
        myotubes = get_myotubes()
    return Nucleis.parse_nucleis(
        roi_coords=np.flip(
            nuclei_pred["coord"].astype(np.int32).transpose(0, 2, 1), axis=2
        ),
        centroids=np.flip(nuclei_pred["points"].astype(np.int16), 1),
        myotubes=myotubes,
        measure_unit=1,
        probs=nuclei_pred["prob"],
    )


def test_nuclei():
    assert Nuclei(**nuclei_info)


def test_myotube():
    assert Myotube(**myotube_info)


def test_myotubes():
    myotubes = get_myotubes()
    assert len(myotubes) == len(myotube_pred)
    assert isinstance(myotubes, Myotubes)


def test_parse_clusters():
    clusters = NucleiClusters.compute_clusters(get_nucleis())
    assert isinstance(clusters, NucleiClusters)


def test_parse_nucleis():
    nucleis = get_nucleis()
    assert len(nucleis) == len(nuclei_pred["coord"])
    assert isinstance(nucleis, Nucleis)


def test_parse_nucleis_empty_myotubes():
    nucleis = get_nucleis(True)
    clusters = NucleiClusters.compute_clusters(nucleis)
    assert len(nucleis) == len(nuclei_pred["coord"])
    assert isinstance(nucleis, Nucleis)
    assert len(clusters) == 0
    assert isinstance(clusters, NucleiClusters)


def test_empty_nucleis():
    nucleis = Nucleis()
    clusters = NucleiClusters.compute_clusters(nucleis)
    assert len(nucleis) == 0
    assert isinstance(nucleis, Nucleis)
    assert len(clusters) == 0
    assert isinstance(clusters, NucleiClusters)


def test_result():
    nucleis = get_nucleis()
    myotubes = get_myotubes()
    myotubes.add_mapping(nucleis.reverse_mapping)
    clusters = NucleiClusters.compute_clusters(nucleis)
    info = InformationMetrics(
        myotubes=myotubes, nucleis=nucleis, nuclei_clusters=clusters
    )
    result = MyoSamInferenceResult(
        myotube_image="myotube_image",
        nuclei_image="nuclei_image",
        information_metrics=info,
    )
    assert result
    assert isinstance(result, MyoSamInferenceResult)
    assert result.model_dump(mode="json")
