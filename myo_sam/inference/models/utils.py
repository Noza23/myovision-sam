import numpy as np
import cv2


def object_overlaps_box(
    points: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Returns True if the point is inside the box. takes points of N objects

    Args:
        points (np.array): N x conts x 2 array of points. (x, y)
        box (tuple[int, int, int, int]): (x, y, w, h) of the box.

    Returns:
        np.ndarray: of shape (N, ) with True if any point of the object
          is inside the box.
    """
    nuclei_inside_myotube = (
        (points[:, :, 0] >= box[0])
        & (points[:, :, 0] <= box[0] + box[2])
        & (points[:, :, 1] >= box[1])
        & (points[:, :, 1] <= box[1] + box[3])
    )
    assert nuclei_inside_myotube.shape == (points.shape[:2])
    return np.any(nuclei_inside_myotube, axis=1)


def object_overlaps_polygon(
    points: np.ndarray,
    target_roi: np.ndarray,
) -> np.ndarray:
    """
    Returns True if the point is inside the polygon. takes points of a
        single object

    Args:
        points (np.array): shaped (conts x 2) or vectorized (conts * 2, )
            array of points
        target_roi (np.array): N x conts x 2 array of points. (x, y)

    Returns:
        np.ndarray: of shape (N, ) with True if any point of the object
          is inside the polygon.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    for point in points:
        if cv2.pointPolygonTest(target_roi, tuple(point), False) > 0:
            return True
    return False
