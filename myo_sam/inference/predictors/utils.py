import cv2
import numpy as np


def remove_redundant_masks(roi_coords: list[np.ndarray]) -> list[np.ndarray]:
    """
    Remove redundant masks from the ROI coordinates.

    Args:
        list of roi_coords. list[(N, 1, 2)]

    Returns:
        list of roi_coords without duplicates.
    """
    boxes = [cv2.minAreaRect(coord) for coord in roi_coords]
    box_points = np.array([cv2.boxPoints(box) for box in boxes])
    centers = np.array([box[0] for box in boxes])
    angles = np.array([box[2] for box in boxes])
    areas = np.array([np.prod(box[1]) for box in boxes])
    boxes_rot = rotate_objects(box_points, angles, centers)[np.argsort(areas)]
    mask = [
        np.where(box_in_box(box, boxes_rot[i:]))[0][-1] + i
        for i, box in enumerate(boxes_rot)
    ]
    return [roi_coords[i] for i in set(mask)]


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
    roi_coords: list[np.ndarray],
    grid: tuple[int, int],
    patch_size: tuple[int, int],
) -> list[np.ndarray]:
    """
    Merge masks at the split points.

    Args:
        roi_coords: list of roi_coords. list[(N, 1, 2)]
        grid: grid of the patches (y_grid, x_grid)
        patch_size: size of the patch (h, w)

    Returns:
        list of roi_coords merged at the split points.
    """
    edges_y = np.arange(0, grid[0] * patch_size[0], patch_size[0])[1:-1]
    edges_x = np.arange(0, grid[1] * patch_size[1], patch_size[1])[1:-1]

    # Merge over x axis
    edge_x_insts = [roi for roi in roi_coords if is_on_x_edge(roi, edges_x)]
    edge_x_merged = merge_over_axis(edge_x_insts, axis=0, threshold=0.85)
    # Merging over x axis
    edge_y_insts = [roi for roi in edge_x_merged if is_on_y_edge(roi, edges_y)]
    edge_y_merged = merge_over_axis(edge_y_insts, axis=1, threshold=0.85)
    return edge_y_merged


def merge_over_axis(
    instances: list[np.ndarray], axis: int, threshold: float = 0.85
) -> list[np.ndarray]:
    """Merge instances over axis."""
    for i, inst in enumerate(instances):
        for j, inst_ref in enumerate(instances[i + 1 :]):
            if is_one_object(inst, inst_ref, threshold=threshold, axis=axis):
                instances[i] = merge_two_contours(inst, inst_ref)
                _ = instances.pop(j)
    return instances


def merge_two_contours(cont1: np.ndarray, cont2: np.ndarray) -> np.ndarray:
    """Merge two contours."""
    conts = np.concatenate([cont1, cont2])
    offset = np.min(conts, axis=0)
    conts -= offset
    w = np.max(conts[:, :, 0]) + 1
    h = np.max(conts[:, :, 1]) + 1
    binary = np.zeros((h, w), dtype=np.uint8)
    binary[conts[:, :, 1], conts[:, :, 0]] = 1
    new_conts = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0][0]
    return new_conts + offset


def is_one_object(
    roi_coord: np.ndarray,
    roi_ref: np.ndarray,
    threshold: float = 0.85,
    axis: int = 0,
) -> bool:
    """Check if the roi_coord and roi_ref is one object based on IoU."""
    roi_i, ref_i = roi_coord[:, axis], roi_ref[:, axis]
    iou = np.intersect1d(roi_i, ref_i).size / np.union1d(roi_i, ref_i).size
    return iou > threshold


def is_on_x_edge(roi_coord: np.ndarray, edges_x: list) -> bool:
    """Check if the roi_coord is on the x edge of the patch."""
    return np.any(roi_coord[:, 0] == edges_x)


def is_on_y_edge(roi_coord: np.ndarray, edges_y: list) -> bool:
    """Check if the roi_coord is on the y edge of the patch."""
    return np.any(roi_coord[:, 1] == edges_y)


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
