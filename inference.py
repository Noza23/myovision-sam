from itertools import zip_longest
from multiprocessing import Pool
import os

import torch
from omegaconf import DictConfig, OmegaConf

from myo_sam.inference.predictors.config import AmgConfig

from myo_sam.inference.pipeline import Pipeline


def pair_images(f1: list[str], f2: list[str]) -> tuple[list[str], list[str]]:
    """Check if the two lists of file names are paired."""
    if not f1 or not f2:
        raise ValueError("No images found.")
    if len(f1) != len(f2):
        raise ValueError(
            "The number of images in the two directories are not the same."
        )

    f1_c = ["_".join(f.split("_")[:-1]) for f in f1]
    f2_c = ["_".join(f.split("_")[:-1]) for f in f2]

    f1_res, f2_res = [], []
    for i, (x, y) in enumerate(zip(sorted(f1_c), sorted(f2_c))):
        fn1 = f1[f1_c.index(x)]
        fn2 = f2[f2_c.index(y)]
        if x != y:
            raise ValueError(f"Missmatch at index {i}: {fn1} and {fn2}.")
        f1_res.append(fn1)
        f2_res.append(fn2)
    return f1_res, f2_res


def main(config: DictConfig, device_id: int, n_devices: int) -> None:
    """Predict in Batch mode."""
    print("> Setting up configuration...", flush=True)
    stardist_pred, myosam_pred = None, None
    myotube_images, nuclei_images = [], []
    general = config["general"]

    if general["myotube_dir"]:
        myotube_images = os.listdir(general["myotube_dir"])
    if general["nuclei_dir"]:
        nuclei_images = os.listdir(general["nuclei_dir"])

    print("> Pairing images...", flush=True)
    if general["myotube_dir"] and general["nuclei_dir"]:
        myotube_images, nuclei_images = pair_images(
            myotube_images, nuclei_images
        )

    pipeline = Pipeline(
        all_contours=True,
        measure_unit=general["measure_unit"],
        _stardist_predictor=stardist_pred,
        _myosam_predictor=myosam_pred,
    )

    if general["stardist_model"]:
        print("> Loading StarDist model...", flush=True)
        pipeline._stardist_predictor.set_model(general["stardist_model"])

    if general["myosam_model"]:
        print("> Loading MyoSam model...", flush=True)
        if general["device"] == "cpu":
            raise ValueError("Running MyoSAM on CPU is not supported.")
        amg_config = AmgConfig.model_validate(config["amg_config"])
        pipeline._myosam_predictor.update_amg_config(amg_config)
        pipeline._myosam_predictor.set_model(
            general["myosam_model"], general["device"] + ":" + str(device_id)
        )

    print("> Starting inference...", flush=True)
    size = max(len(myotube_images), len(nuclei_images))
    step = size // n_devices
    start = device_id * step
    end = start + step if device_id != n_devices - 1 else size
    myotube_images = myotube_images[start:end]
    nuclei_images = nuclei_images[start:end]

    for myo, nuclei in zip_longest(myotube_images, nuclei_images):
        print(f"> Processing {myo} and {nuclei}...", flush=True)
        pipeline.set_myotube_image(
            os.path.join(general["myotube_dir"], myo), myo
        )
        pipeline.set_nuclei_image(
            os.path.join(general["nuclei_dir"], nuclei), nuclei
        )
        result = pipeline.execute()

        result_fn = "_".join(myo.split("_")[:-1]) + ".json"
        print(f"> Saving result to {result_fn}...", flush=True)
        result.save(os.path.join(general["output_dir"], result_fn))
        pipeline.clear_cache()


if __name__ == "__main__":
    n_devices = torch.cuda.device_count()
    config = OmegaConf.load("inference.yaml")
    print(f"> Found {n_devices} CUDA devices.", flush=True)
    # Spawn a process for each device
    print("> Starting Batch Inference...", flush=True)
    if n_devices <= 1:
        main(config, 0, 1)
    else:
        with Pool(n_devices) as p:
            p.starmap(main, [(config, i, n_devices) for i in range(n_devices)])
    print("> Done.", flush=True)
