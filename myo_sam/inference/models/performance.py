from pydantic import BaseModel, Field

from . import MyoObject


class PerformanceMetrics(BaseModel):
    """The performance metrics of a MyoSam inference."""

    accuracy: float = Field(..., example=0.98)
    precision: float = Field(..., example=0.98)
    recall: float = Field(..., example=0.98)
    f1_score: float = Field(..., example=0.98)
    confusion_matrix: list = Field(..., example=[[0.98, 0.02], [0.01, 0.99]])

    @classmethod
    def compute_performance(
        cls, prediction: MyoObject, ground_truth: MyoObject
    ) -> "PerformanceMetrics":
        return PerformanceMetrics()
