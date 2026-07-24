import numpy as np
from scipy import ndimage


# Fakes a relief height map from a single photo. We have no real depth, so we
# lean on a simple fact: raised detail catches the light and reads brighter
# than the metal right next to it, even when the whole area sits in shadow.
# Subtracting a blurred copy of the luminance cancels the lighting and leaves
# roughly the relief.
class SignedRelief:

    def __init__(self, field_sigma=20.0, blur_sigma=1.0, broaden_sigma=2.0,
                 lo_pct=4.0, hi_pct=99.0):
        self.field_sigma = field_sigma       # blur that stands in for local lighting
        self.blur_sigma = blur_sigma         # tiny denoise blur before anything else
        self.broaden_sigma = broaden_sigma   # spreads the estimate over whole features

        # percentiles mapped to height 0 and 1
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct

    # Build the height map for one face. Takes the float RGB image and the coin
    # mask, gives back a height map in [0, 1] that's zero outside the coin.
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
        h = np.clip((relief - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return (h * coin_mask).astype(np.float32)
