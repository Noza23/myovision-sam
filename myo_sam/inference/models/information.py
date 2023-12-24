from pydantic import BaseModel, Field, computed_field

from .base import Myotubes, Nucleis, NucleiClusters
from collections import Counter


class InformationMetrics(BaseModel):
    """The information metrics of a MyoSam inference."""

    myotubes: Myotubes = Field(description="The myotubes.")
    nucleis: Nucleis = Field(description="The nucleis.")
    nuclei_clusters: NucleiClusters = Field(description="nuclei clusters.")

    @computed_field  # type: ignore[misc]
    @property
    def total_fusion_index(self):
        return (
            self.nucleis.num_nuclei_inside_myotube / self.nucleis.num_myoblasts
        )

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei_per_instance(self) -> dict[int, int]:
        """The number of nuclei per myotube instance"""
        counter = Counter([n.myotube_id for n in self.nucleis.myo_objects])
        myotube_ids = [m.identifier for m in self.myotubes.myo_objects]
        return {m_id: counter[m_id] for m_id in myotube_ids}

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei_clusters(self) -> int:
        """The number of nuclei clusters."""
        return len(self.nuclei_clusters)

    @computed_field  # type: ignore[misc]
    @property
    def num_nuclei_per_nuclei_cluster(self) -> dict[int, int]:
        """The number of nuclei per nuclei cluster."""
        return dict(
            Counter([n.cluster_id for n in self.nuclei_clusters.clusters])
        )

    @computed_field  # type: ignore[misc]
    @property
    def total_myotubes(self) -> int:
        """The number of myotubes."""
        return len(self.myotubes)

    @computed_field  # type: ignore[misc]
    @property
    def total_myotubes_area(self) -> int:
        """The total area of the myotubes."""
        return self.myotubes.area

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis(self) -> int:
        """The number of nuclei."""
        return len(self.nucleis)

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis_area(self) -> int:
        """The total area of the nuclei."""
        return self.nucleis.area

    @computed_field  # type: ignore[misc]
    @property
    def total_myoblasts(self) -> int:
        """The number of myoblasts."""
        return self.nucleis.num_myoblasts

    @computed_field  # type: ignore[misc]
    @property
    def total_myoblasts_area(self) -> int:
        """The total area of the myoblasts."""
        return self.nucleis.myoblasts_area

    @computed_field  # type: ignore[misc]
    @property
    def total_nucleis_inside_myotubes_area(self) -> int:
        """The total area of the nuclei inside myotubes."""
        return self.nucleis.nucleis_inside_myotubes_area
