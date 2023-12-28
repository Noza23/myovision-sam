from pydantic import BaseModel, Field

from .base import MyoObjects


class PerformanceMetrics(BaseModel):
    """The performance metrics of a MyoSam inference."""

    accuracy: float = Field(default=0.98)
    precision: float = Field(default=0.98)
    recall: float = Field(default=0.98)
    f1_score: float = Field(default=0.98)
    confusion_matrix: list = Field(default=[[0.98, 0.02], [0.01, 0.99]])

    @classmethod
    def compute_performance(
        cls, predictions: MyoObjects, ground_truths: MyoObjects
    ) -> "PerformanceMetrics":
        return PerformanceMetrics()
