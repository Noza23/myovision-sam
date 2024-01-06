import numpy as np
import hashlib


def hash_array(arr: np.ndarray):
    """Hash a numpy array for caching."""
    hash = hashlib.sha256(arr.tobytes())
    return hash.hexdigest()


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
