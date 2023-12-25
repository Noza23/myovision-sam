from functools import cached_property
from typing import Union
import math
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field, computed_field
import cv2
import numpy as np

from .utils import object_overlaps_box, object_overlaps_polygon


class MyoObject(BaseModel):
    """Base Class for MyoObjects: Myotubes and Nucleis."""

    identifier: int = Field(description="Identifier of the myoobject.")
    # roi_coords are computed using cv2.findContours conts[0].tolist()
    # cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    roi_coords: list[list[int]] = Field(description="ROI boundaries")  # (x, y)
    measure_unit: float = Field(description="Measure unit of the myoobject.")

    @cached_property
    def roi_coords_np(self) -> np.ndarray:
        """ROI coordinates as a numpy array. (N, 1, 2)"""
        return np.array(self.roi_coords)[:, None, :]

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> float:
        """Area of the myoobject."""
        return cv2.contourArea(self.roi_coords_np) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def convex_area(self) -> float:
        """Convex area of the myoobject."""
        return cv2.contourArea([self.convex_hull]) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def solidity(self) -> float:
        """Solidity of the myoobject."""
        return self.area / self.convex_area

    @computed_field  # type: ignore[misc]
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the myoobject."""
        return self.elipse[1][0] / self.elipse[1][1]

    @computed_field  # type: ignore[misc]
    @property
    def roundness(self) -> float:
        """Roundness of the myoobject."""
        return 4 * self.area / (math.pi * self.elipse[1][0] ** 2)

    @computed_field  # type: ignore[misc]
    @property
    def perimeter(self) -> float:
        """Perimeter of the myoobject."""
        return cv2.arcLength(self.roi_coords_np, True) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def max_feret_diameter(self):
        """Feret's diameter of the myoobject."""
        return max(self.feret_bound_box)

    @computed_field  # type: ignore[misc]
    @property
    def min_feret_diameter(self):
        """Feret's diameter of the myoobject."""
        return min(self.feret_bound_box)

    @computed_field  # type: ignore[misc]
    @property
    def circularity(self) -> float:
        """Circularity of the myoobject."""
        return 4 * math.pi * self.area / (self.perimeter**2)

    @computed_field  # type: ignore[misc]
    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid of the myoobject. (x, y)"""
        return self.elipse[0]

    @computed_field  # type: ignore[misc]
    @property
    def convex_hull(self) -> list[list[list[int]]]:
        """Convex hull of the myoobject."""
        return cv2.convexHull(self.roi_coords_np)

    @cached_property
    def feret_bound_box(self) -> list[int]:
        """Feret's bounding box of the myoobject."""
        _, eig_vecs = cv2.PCACompute(self.roi_coords_np, mean=None)
        coords = np.matmul(eig_vecs, np.array(self.roi_coords_np).T)
        return cv2.boundingRect(coords.T)[2:]

    @cached_property
    def elipse(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """
        Elipse of the myoobject.

        Returns:
            ((center_x, center_y), (major_axis, minor_axis), angle)
        """
        return cv2.fitEllipse(self.roi_coords_np)


class Myotube(MyoObject):
    rle_mask: list[int] = Field(description="RLE mask of the myoobject.")
    rgb_repr: list[list[int]] = Field(description="RGB representation")

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> float:
        """Area of the myoobject."""
        return sum(self.rle_mask[1::2]) * self.measure_unit

    @computed_field  # type: ignore[misc]
    @property
    def rgb_min(self) -> tuple:
        """Minimum intensity of the myotube per channel."""
        return tuple(min(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def rgb_max(self) -> tuple:
        """Maximum intensity of the myotube per channel."""
        return tuple(max(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def rgb_mean(self) -> tuple:
        """Mean intensity of the myotube per channel."""
        return tuple(statistics.mean(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def rgb_median(self) -> tuple:
        """Median intensity of the myotube per channel."""
        return tuple(statistics.median(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def rgb_mode(self) -> tuple:
        """Mode intensity of the myotube per channel."""
        return tuple(statistics.mode(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def rgb_std(self) -> tuple:
        """Standard deviation of the myotube per channel."""
        return tuple(statistics.stdev(c) for c in zip(*self.rgb_repr))

    @computed_field  # type: ignore[misc]
    @property
    def integrated_density_rgb(self) -> tuple:
        """Integrated density of the myotube per channel."""
        return tuple(sum(c) for c in zip(*self.rgb_repr))


class Nuclei(MyoObject):
    """A detected nuclei."""

    myotube_ids: list[Union[int, None]] = Field(
        description="Identifer of the myotubes the nuclei belongs to."
    )


class MyoObjects(BaseModel):
    """Base class for myotubes and myoblasts and other detected objects."""

    myo_objects: list[MyoObject] = Field("List of myoobjects.")

    def __len__(self) -> int:
        return len(self.myo_objects)

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> float:
        """Area of the myoobjects."""
        return sum([m.area for m in self.myo_objects])


class Myotubes(MyoObjects):
    """The myotubes of a MyoSam inference."""

    myo_objects: list[Myotube] = Field("List of myotubes.")

    def get_myotube_by_id(self, id: int) -> Myotube:
        return [m for m in self.myo_objects if m.identifier == id][0]


class Nucleis(MyoObjects):
    """The nucleis of a MyoSam inference."""

    myo_objects: list[Nuclei] = Field("List of nucleis.")
    mapp: dict[int, list[int]] = Field(
        description="Mapping of the nucleis to the myotubes."
    )
    mapp_reverse: dict[int, list[int]] = Field(
        description="Mapping of the myotubes to the nucleis."
    )

    @computed_field  # type: ignore[misc]
    @property
    def num_myoblasts(self) -> int:
        """Number of myoblasts."""
        return len([m for m in self.myo_objects if not m.myotube_ids])

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei_inside_myotube(self) -> int:
        """Number of myotubes."""
        return len(self) - self.num_myoblasts

    @computed_field  # type: ignore[misc]
    @property
    def myoblasts_area(self) -> float:
        """Area of the myoblasts."""
        return sum([m.area for m in self.myo_objects if not m.myotube_ids])

    @computed_field  # type: ignore[misc]
    @property
    def nucleis_inside_myotubes_area(self) -> float:
        """Area of the myotubes."""
        return self.area - self.myoblasts_area

    @classmethod
    def parse_nucleis(
        cls, roi_coords: np.ndarray, myotubes: Myotubes
    ) -> "Nucleis":
        """
        Parses the nucleis from the roi_coords and myotubes.
        Args:
            roi_coords (np.array): N x n_conts x 2 array of points. (x, y)
            myotubes (Myotubes): The myotubes.
        """
        # np.flip(myoblast_rois.transpose(0, 2, 1), axis=2).astype(np.uint16)
        mapp = defaultdict(list)
        mapp_reverse = defaultdict(list)
        for myotube in myotubes.myo_objects:
            box = cv2.boundingRect(myotube.roi_coords_np)
            msk = np.where(object_overlaps_box(roi_coords, box))[0]
            nucleis = np.apply_along_axis(
                object_overlaps_polygon,
                -1,
                roi_coords[msk].reshape(len(msk), -1),
                myotube.roi_coords_np,
            )
            for i in msk[nucleis]:
                mapp[i].append(myotube.identifier)
                mapp_reverse[myotube.identifier].append(i)

        nucleis = [
            Nuclei(
                identifier=i,
                roi_coords=coords,
                measure_unit=1,
                myotube_id=mapp[i],
            )
            for i, coords in enumerate(roi_coords)
        ]
        return cls(myo_objects=nucleis, mapp=mapp, mapp_reverse=mapp_reverse)


class NucleiCluster(MyoObjects):
    """A detected nuclei cluster."""

    cluster_id: int = Field("Cluster identifier")
    myo_objects: list[Nuclei] = Field(description="List of nucleis.")

    @classmethod
    def compute_clusters(cls, nucleis: Nucleis) -> "NucleiCluster":
        """Computes the clusters of nucleis."""

        raise NotImplementedError


class NucleiClusters(BaseModel):
    """The nuclei clusters of a MyoSam inference."""

    clusters: list[NucleiCluster] = Field(description="List of clusters.")

    def __len__(self) -> int:
        return len(self.clusters)
