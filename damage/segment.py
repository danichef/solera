from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# A single coin face cut out of a scan.
@dataclass
class Face:
    image: np.ndarray
    coin_mask: np.ndarray
    bbox: tuple
    name: str


# Pull the coin faces out of a scan shot on a white background. Anything darker
# than white_cutoff counts as foreground; blobs bigger than min_area_frac of the
# image become faces. Returns a list of Face objects ordered left to right.
def segment_faces(rgb_uint8, white_cutoff=250, min_area_frac=0.02):
    grey = rgb_uint8.mean(axis=2)
    fg = ndimage.binary_fill_holes(grey < white_cutoff)

    labels, n = ndimage.label(fg)
    if n == 0:
        return []

    areas = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    min_area = min_area_frac * fg.size
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]

    def left_edge(label):
        return np.where(labels == label)[1].min()

    faces = []
    for i, label in enumerate(sorted(keep, key=left_edge)):
        blob = labels == label
        ys, xs = np.where(blob)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        faces.append(Face(
            image=rgb_uint8[y0:y1, x0:x1].astype(np.float32) / 255.0,
            coin_mask=blob[y0:y1, x0:x1].astype(np.float32),
            bbox=(x0, y0, x1, y1),
            name=f"face{i}",
        ))
    return faces
