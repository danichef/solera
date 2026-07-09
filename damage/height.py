from __future__ import annotations

import numpy as np
from scipy import ndimage


# Estimates a relief height map from a single photo. There is no real depth
# data, so brightness relative to the local neighbourhood stands in for
# height: raised detail catches the light and reads brighter than the metal
# right next to it, even when the whole region is in shadow.
class SignedRelief:

    # @params field_sigma: blur radius that defines the "local lighting field"
    # @params blur_sigma: small denoise blur before anything else
    # @params broaden_sigma: blur that spreads the estimate over whole features
    # @params lo_pct / hi_pct: percentiles mapped to height 0 and 1
    def __init__(self, field_sigma: float = 20.0, blur_sigma: float = 1.0,
                 broaden_sigma: float = 2.0,
                 lo_pct: float = 4.0, hi_pct: float = 99.0):
        self.field_sigma = field_sigma
        self.blur_sigma = blur_sigma
        self.broaden_sigma = broaden_sigma
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct

    # Builds the height map for one face.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of coin pixels
    # @output float height map in [0, 1], zero outside the coin
    def estimate(self, image, coin_mask):
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        if self.blur_sigma > 0:
            lum = ndimage.gaussian_filter(lum, self.blur_sigma)

        relief = lum - ndimage.gaussian_filter(lum, self.field_sigma)
        if self.broaden_sigma > 0:
            relief = ndimage.gaussian_filter(relief, self.broaden_sigma)

        inside = coin_mask > 0.5
        if not inside.any():
            return np.zeros_like(relief, dtype=np.float32)

        lo = np.percentile(relief[inside], self.lo_pct)
        hi = np.percentile(relief[inside], self.hi_pct)
        height = np.clip((relief - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return (height * coin_mask).astype(np.float32)
