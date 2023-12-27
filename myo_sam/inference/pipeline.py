# Flip stardist coordinats
# np.flip(myoblast_rois.transpose(0, 2, 1), axis=2).astype(np.int32)

from pydantic import BaseModel


class Pipeline(BaseModel):
    """The pipeline of a MyoSam inference."""

    pass
