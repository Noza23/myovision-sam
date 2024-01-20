import cv2
import numpy as np


def remove_redundant_masks(roi_coords: list[np.ndarray]) -> list[np.ndarray]:
    """
    Remove redundant masks from the ROI coordinates.

    Args:
        list of roi_coords.

    Returns:
        list of roi_coords without duplicates.
    """
    boxes = [cv2.minAreaRect(coord) for coord in roi_coords]
    box_points = np.array([cv2.boxPoints(box) for box in boxes])
    centers = np.array([box[0] for box in boxes])
    angles = np.array([box[2] for box in boxes])
    areas = np.array([np.prod(box[1]) for box in boxes])
    boxes_rot = rotate_objects(box_points, angles, centers)[np.argsort(areas)]
    mask = [np.where(box_in_box(box, boxes_rot))[0][0] for box in boxes_rot]
    return [roi_coords[i] for i in mask]


def box_in_box(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Check if a box is inside another box.

    Args:
        box: (4, 2) array of points
        boxes: (N, 4, 2) array of points
    """
    return np.all((box[0] >= boxes[:, 0]) & (box[-1] <= boxes[:, -1]), axis=1)


def remove_disconnected_masks(roi_coords: np.ndarray):
    """Remove disconnected masks."""
    # Background is always connected
    mask = [cv2.connectedComponents(coord)[0] > 2 for coord in roi_coords]
    return roi_coords[mask]


def merge_masks_at_splitponits(
    roi_coords: np.ndarray, grid_size: tuple[int, int]
):
    """Merge masks at the split points."""

    # TODO: Find instances on edges
    # TODO: Find Counterparts and merge them

    # TODO: Check if any instance  coordinate touches the edge of the image
    # TODO: Check instance vs instance on x axis iou > 85%
    # TODO: Check instance vs isntance on y axis iou > 85%

    # TODO: Return new merged coordinates
    pass


def rotate_objects(
    points_array: np.ndarray, angles: np.ndarray, centers_array: np.ndarray
) -> np.ndarray:
    """
    Rotate multiple sets of points by given angles and centers.

    Args:
        points_array: array of shape (N, 2)
        angles: array of shape (N,)
        centers_array: array of shape (N, 2)

    Returns:
        sorted rotated points array of shape (B, N, 2)
        (In case of a rect: bottom_left, top_left, top_right, bottom_right)
    """
    rad_angles = np.radians(angles)
    rot_mat = np.array(
        [
            [np.cos(rad_angles), -np.sin(rad_angles)],
            [np.sin(rad_angles), np.cos(rad_angles)],
        ]
    ).transpose((2, 0, 1))
    ps_rot = (
        np.matmul(points_array - centers_array[:, None, :], rot_mat)
        + centers_array[:, None, :]
    )
    ps_rot = np.array([b[np.lexsort((b[:, 1], b[:, 0]))] for b in ps_rot])
    return ps_rot
