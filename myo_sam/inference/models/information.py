from pydantic import BaseModel, Field

from . import Myoblast, MyoObject


# class Nuclei(Myoblast):
#     """A detected nuclei should be inside a myotube by at least 95%."""
#     myotube_id: int = Field(
#         description="Identifer of the myotube the nuclei belongs to."
#     )
#     @classmethod
#     def compute_nuclei(cls, myotubes: MyoObjects, myoblasts: MyoObjects) -> "Nuclei":
#         # Binarize myotubes on full scale ()
#         pass


class NucleisInMyotube(BaseModel):
    """A detected nuclei should be inside a myotube by at least 95%."""

    myotube_id: int = Field(
        description="Identifer of the myotube the nuclei belongs to."
    )
    nucleis: list[Myoblast] = Field(description="The nucleis of a myotube.")

    @classmethod
    def identify_nucleis(
        cls, myotube: MyoObject, myoblasts: list[MyoObject]
    ) -> "NucleisInMyotube":
        nucleis = [m for m in myoblasts if m.centroid in myotube.roi_coords]
        return cls(myotube_id=myotube.identifier, nucleis=nucleis)

    # def identify_clusters(self) -> list[list[MyoObject]]:
    #     for nuclei in self.nucleis:
    #         pass
    #         #  Start from first myoblast and build up a cluster iteratively
    #     pass


# Clusters of Nuclei....


# class NucleiCluster(MyoObjects):
#     """A detected nuclei cluster."""
#     @classmethod
#     def compute_clusters(cls, nucleis: Nucleis) -> "NucleiCluster":
#         pass

# class NucleiClusters(BaseModel):
#     """The nuclei clusters of a MyoSam inference."""
#     nuclei_clusters: list[NucleiCluster] = Field(
#         description="The nuclei clusters of a MyoSam inference."
#     )
#     def __len__(self) -> int:
#         return len(self.nuclei_clusters)


class InformationMetrics(BaseModel):
    """The information metrics of a MyoSam inference."""

    # Myotubes
    myotube_count: int = Field(description="The number of myotubes.")
    myotube_areas: list[int] = Field(
        description="The areas of the myotubes, sorted by the identifier.",
    )

    # Myoblasts: Everywhere in the image
    myoblast_count: int = Field(description="The number of myoblasts.")
    myoblast_areas: list[int] = Field(
        description="The areas of the myoblasts, sorted by the identifier.",
    )

    # Nuclei: Myoblasts touching the myotube by at least 95%
    nuclei_count: int = Field(description="The number of nuclei.")
    nuclei_areas: list[int] = Field(
        description="The areas of the nuclei, sorted by the identifier.",
    )

    # Nuclei: Nucleis overlapping by at least 10%
    nuclei_clusters_count: int = Field(
        description="The number of nuclei clusters."
    )

    # @classmethod
    # def compute_metrics(
    #     cls, myotubes: MyoObjects, myoblasts: MyoObjects
    # ) -> "InformationMetrics":
    #     nucleis: Nucleis = cls.compute_nucleis(myotubes, myoblasts)

    #     pass
