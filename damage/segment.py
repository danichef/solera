from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# One coin face cut out of a scan: the cropped image, its coin mask, the crop
# box in the original scan, and a name for output files.
@dataclass
class Face:
    image: np.ndarray
    coin_mask: np.ndarray
    bbox: tuple
    name: str


# Finds the coin faces in a scan photographed on a white background. Everything
# darker than the cutoff is foreground; connected blobs above the minimum area
# become faces, ordered left to right.
# @params rgb_uint8: the scan as a uint8 RGB array
# @params white_cutoff: gray level above which a pixel counts as background
# @params min_area_frac: smallest blob to keep, as a fraction of the image
# @output list of Face objects, left to right
def segment_faces(rgb_uint8: np.ndarray,
                  white_cutoff: int = 250,
                  min_area_frac: float = 0.02) -> list[Face]:
    gray = rgb_uint8.mean(axis=2)
    foreground = ndimage.binary_fill_holes(gray < white_cutoff)

    labels, count = ndimage.label(foreground)
    if count == 0:
        return []

    areas = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
    min_area = min_area_frac * foreground.size
    keep = [i + 1 for i, area in enumerate(areas) if area >= min_area]

    def left_edge(label):
        return np.where(labels == label)[1].min()

    faces = []
    for index, label in enumerate(sorted(keep, key=left_edge)):
        component = labels == label
        ys, xs = np.where(component)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        faces.append(Face(
            image=rgb_uint8[y0:y1, x0:x1].astype(np.float32) / 255.0,
            coin_mask=component[y0:y1, x0:x1].astype(np.float32),
            bbox=(x0, y0, x1, y1),
            name=f"face{index}",
        ))
    return faces
