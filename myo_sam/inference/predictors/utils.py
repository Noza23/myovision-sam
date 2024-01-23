from typing import Optional

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
    order_i = np.argsort(areas)
    boxes_rot = rotate_objects(box_points, angles, centers)[order_i]
    roi_coords = [roi_coords[i] for i in order_i]
    # 5 // 5 esbox [-1]
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


def split_image_into_patches(
    image: np.ndarray, patch_size: tuple[int, int]
) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Split an image into patches of a given size."""
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    if patch_height > height or patch_width > width:
        return (0, 0), [image]

    patches = []
    width_reminder = width % patch_width
    height_reminder = height % patch_height
    width_range = [*range(0, width, patch_width)]
    height_range = [*range(0, height, patch_height)]

    # if reminder less than quarter of patch-size, merge it to the last patch
    if width_reminder < patch_size[0] / 4:
        width_range[-1] += width_reminder
    else:
        width_range.append(width)
    # if reminder less than quarter of patch-size, merge it to the last patch
    if height_reminder < patch_size[1] / 4:
        height_range[-1] += height_reminder
    else:
        height_range.append(height)

    # grid is needed for later reconstruction
    grid = (len(height_range) - 1, len(width_range) - 1)

    for i in range(len(height_range) - 1):
        for j in range(len(width_range) - 1):
            left, right = width_range[j], width_range[j + 1]
            lower, upper = height_range[i], height_range[i + 1]
            patch = image[lower:upper, left:right, ...]
            patches.append(patch)
    return grid, patches


def remove_disconnected_masks(roi_coords: np.ndarray):
    """Remove disconnected masks."""
    # Background is always connected
    mask = [cv2.connectedComponents(coord)[0] > 2 for coord in roi_coords]
    return roi_coords[mask]


def merge_masks_at_splitponits(
    roi_coords: list[np.ndarray],
    grid: tuple[int, int],
    patch_size: tuple[int, int],
    iou_threshold: float = 0.85,
    max_offset: int = 1,
) -> tuple[list[np.ndarray], list[list]]:
    """
    Merge masks at the split points.

    Args:
        roi_coords: list of roi_coords. list[(N, 1, 2)]
        grid: grid of the patches (y_grid, x_grid)
        patch_size: size of the patch (h, w)

    Returns:
        list of roi_coords merged at the split points.
        list of list of ids containing the ids of the merged masks.
    """
    ids = np.arange(len(roi_coords))[:, None].tolist()
    if grid[1] > 1:
        axis = 0
        edges_x = np.arange(0, grid[1] * patch_size[1], patch_size[1])[1:]
        for edge in edges_x:
            edges = np.arange(edge - max_offset, edge + max_offset + 1)
            is_on_edge_x = [is_on_edge(roi, edges, axis) for roi in roi_coords]
            ids_on_edge_x = [id_x for i, id_x in zip(is_on_edge_x, ids) if i]
            ids = [id_x for i, id_x in zip(is_on_edge_x, ids) if not i]
            edge_x_insts = [ro for i, ro in zip(is_on_edge_x, roi_coords) if i]
            edge_x_merged, merged_ids = merge_over_axis(
                edge_x_insts, axis, edges, iou_threshold, ids_on_edge_x
            )
            # Merge not on edge with merged edge instances
            ids.extend(merged_ids)
            roi_coords = [
                roi for i, roi in zip(is_on_edge_x, roi_coords) if not i
            ]
            roi_coords.extend(edge_x_merged)

    if grid[0] > 1:
        axis = 1
        edges_y = np.arange(0, grid[0] * patch_size[0], patch_size[0])[1:]
        for edge in edges_y:
            edges = np.arange(edge - max_offset, edge + max_offset + 1)
            is_on_edge_y = [is_on_edge(roi, edges, axis) for roi in roi_coords]
            ids_on_edge_y = [id_x for i, id_x in zip(is_on_edge_y, ids) if i]
            ids = [id_x for i, id_x in zip(is_on_edge_y, ids) if not i]
            edge_y_insts = [ro for i, ro in zip(is_on_edge_y, roi_coords) if i]
            edge_y_merged, merged_ids = merge_over_axis(
                edge_y_insts, axis, edges, iou_threshold, ids_on_edge_y
            )
            # Merge not on edge with merged edge instances
            ids.extend(merged_ids)
            roi_coords = [
                roi for i, roi in zip(is_on_edge_y, roi_coords) if not i
            ]
            roi_coords.extend(edge_y_merged)
    return roi_coords, ids


def merge_over_axis(
    instances: list[np.ndarray],
    axis: int,
    edges: np.ndarray,
    threshold: float = 0.85,
    ids: Optional[list[list[int]]] = None,
) -> tuple[list[np.ndarray], list[list]]:
    """Merge instances over axis."""
    if not ids:
        ids = [[i] for i in range(len(instances))]

    for i, (id_i, inst) in enumerate(zip(ids, instances)):
        for j, (id_j, inst_ref) in enumerate(
            zip(ids[i + 1 :], instances[i + 1 :])
        ):
            if is_one_object(inst, inst_ref, edges, threshold, axis=axis):
                instances[i] = merge_two_contours(inst, inst_ref)
                ids[i] = id_i + id_j
                instances.pop(i + j + 1), ids.pop(i + j + 1)
                j -= 1
    return instances, ids


def is_on_edge(roi_coord: np.ndarray, edges_x: list, axis: int) -> bool:
    """Check if the roi_coord is on the x edge of the patch."""
    return np.any(roi_coord[:, :, 1 - axis] == edges_x)


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
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0][0]
    return new_conts + offset


def is_one_object(
    roi_coord: np.ndarray,
    roi_ref: np.ndarray,
    edges: np.ndarray,
    threshold: float = 0.85,
    axis: int = 0,
) -> bool:
    """Check if the roi_coord and roi_ref is one object based on IoU."""
    roi_i = roi_coord[:, :, axis][
        np.any(roi_coord[:, :, 1 - axis] == edges, axis=1)
    ]
    ref_i = roi_ref[:, :, axis][
        np.any(roi_ref[:, :, 1 - axis] == edges, axis=1)
    ]
    iou = np.intersect1d(roi_i, ref_i).size / (
        np.union1d(roi_i, ref_i).size + 1e-6
    )
    return iou > threshold


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
