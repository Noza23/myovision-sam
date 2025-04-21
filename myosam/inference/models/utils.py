import numpy as np
import cv2
from collections import OrderedDict


def object_overlaps_box(
    points: np.ndarray, box: tuple[int, int, int, int]
) -> bool:
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
) -> bool:
    """
    Returns True if the point is inside the polygon. takes points of a
        single object

    Args:
        points (np.array): shaped (conts x 2) or vectorized (conts * 2, )
            array of points (x, y) int16
        target_roi (np.array): N x conts x 2 array of points. (x, y) int32

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


def object_overlaps_by_perc(
    points: np.ndarray,
    target_box: tuple[int, int, int, int],
    target_roi: np.ndarray,
    percentage: float = 0.95,
) -> bool:
    """
    Returns True if one polygon overlaps the other by a certain percentage.

    Args:
        points (np.array): n_conts x 1 x 2 array of points. (x, y)
        target_box (tuple[int, int, int, int]): (x, y, w, h) of the box.
        target_roi (np.array): n_conts x 1 x 2 array of points. (x, y) int32
        percentage (float): percentage of the object that must be inside the
            polygon for it to be considered inside.
    """
    source = np.zeros(target_box[2:4][::-1], dtype=np.uint8)
    target = np.zeros(target_box[2:4][::-1], dtype=np.uint8)
    target_roi_lower = target_roi - target_box[:2]
    source_roi_lower = points - target_box[:2]
    _ = cv2.fillPoly(target, [target_roi_lower], 255)
    _ = cv2.fillPoly(source, [source_roi_lower], 255)
    return cv2.bitwise_and(source, target).sum() / source.sum() >= percentage


def vec_to_sym_matrix(vec: np.ndarray, ms: int) -> np.ndarray:
    """
    Takes vectorized lower triangular matrix and returns symmetric matrix

    Args:
        vec (np.ndarray): vectorized lower triangular matrix (n, )
        ms (int): symmetric matrix size
    """
    assert vec.shape[0] == ms * (ms - 1) / 2
    sym_matrix = np.zeros((ms, ms), dtype=bool)
    sym_matrix[np.triu_indices(ms, 1)] = vec
    sym_matrix = sym_matrix | sym_matrix.T
    np.fill_diagonal(sym_matrix, True)
    return sym_matrix


def rois2coords(rois: OrderedDict) -> list[np.ndarray]:
    """
    Converts a ordered dictionary of ROIs into a list of coordinates.

    Args:
        rois (OrderedDict): Dictionary of ROIs.

    Returns:
        list: List of coordinates (numpy arrays).
    """

    coords_lst = []
    for k, roi in rois.items():
        coords = np.round(np.stack((roi["x"], roi["y"]), axis=1)).astype(
            np.int32
        )[:, None, :]
        coords_lst.append(coords)
    return coords_lst


def coord2cont(coord: np.ndarray, im_shape: tuple[int, int]) -> np.ndarray:
    """
    Converts coordinates of a binary mask contour into a contour of a binary mask.

    Args:
        coord (np.ndarray): Contour (numpy array).
        im_shape (tuple): Shape of the image.

    Returns:
        np.ndarray: Contour (numpy array).
    """

    contour = np.zeros(im_shape, dtype=np.uint8).copy()
    cv2.drawContours(contour, [coord], -1, (1), thickness=1)
    return contour


def coord2mask(coord: np.ndarray, im_shape: tuple[int, int]) -> np.ndarray:
    """
    Converts coordinates of a binary mask contour into a binary mask.

    Args:
        coord (np.ndarray): Contour (numpy array).
        im_shape (tuple): Shape of the image.

    Returns:
        np.ndarray: Binary mask (numpy array).
    """

    contour = np.zeros(im_shape, dtype=np.uint8).copy()
    cv2.drawContours(contour, [coord], -1, (1), thickness=-1)
    return contour


def mask2cont(mask: np.ndarray, im_shape: tuple[int, int]) -> np.ndarray:
    """
    Converts a binary mask into a contour.

    Args:
        mask (np.ndarray): Binary mask (numpy array).
        im_shape (tuple): Shape of the image.

    Returns:
        np.ndarray: Contour (numpy array).
    """

    mask_np = mask.astype(np.uint8)
    coords, _ = cv2.findContours(
        mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cont = np.zeros(im_shape, dtype=np.uint8).copy()
    cv2.drawContours(cont, coords, -1, (1))
    return cont
