import numpy as np
from scipy import ndimage

from .base import DamageFilter, DamageResult


# band-limited noise around a circle, for roughening chip outlines
def _angular_noise(rng, n, low=2.0, high=9.0):
    k = np.fft.rfftfreq(n, d=1.0 / n)
    F = np.fft.rfft(rng.standard_normal(n))
    band = np.exp(-((k - 0.5 * (low + high)) / max(0.5 * (high - low), 1e-6)) ** 2)
    band[0] = 0.0
    prof = np.fft.irfft(F * band, n)
    return (prof / max(prof.std(), 1e-6)).astype(np.float32)


# Damages the coin's outline. Small nibbles come off the rim wherever a ring of
# thresholded noise peaks, and now and then a big circular flan clip is taken.
# Cut edges get a dark contour so they read as broken metal rather than erased,
# and the coin mask is shrunk to the new shape.
class ChipFilter(DamageFilter):

    # @params amplitude: depth of the rim nibbles, relative to the radius
    # @params threshold: noise level a peak must clear to become a chip
    # @params low_weight / high_weight / low_cut / high_lo / high_hi: the two
    #         frequency bands of the rim noise ring
    # @params sharpness: exponent shaping the bite profile
    # @params waviness: gentle irregularity over the whole rim
    # @params n_angles: angular resolution of the rim profile
    # @params big_chip_prob: chance of a large flan clip
    # @params big_chip_depth / big_chip_radius: clip size ranges, in radii
    # @params big_chip_rough: roughness of the clip outline
    # @params second_chip_frac: chance a clip gets a partner opposite it
    # @params edge_shade: darkness of the shading band along the cut
    # @params edge_width_frac / shadow_margin_frac: geometry of that shading
    def __init__(self,
                 amplitude=0.07,
                 threshold=0.48,
                 low_weight=1.0,
                 high_weight=0.32,
                 low_cut=3.5,
                 high_lo=9.0,
                 high_hi=20.0,
                 sharpness=0.7,
                 waviness=0.010,
                 n_angles=1440,
                 big_chip_prob=0.0,
                 big_chip_depth=(0.06, 0.15),
                 big_chip_radius=(0.45, 0.95),
                 big_chip_rough=0.03,
                 second_chip_frac=0.30,
                 edge_shade=0.55,
                 edge_width_frac=0.030,
                 shadow_margin_frac=0.045):
        self.amplitude = amplitude
        self.threshold = threshold
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.low_cut = low_cut
        self.high_lo = high_lo
        self.high_hi = high_hi
        self.sharpness = sharpness
        self.waviness = waviness
        self.n_angles = n_angles
        self.big_chip_prob = big_chip_prob
        self.big_chip_depth = big_chip_depth
        self.big_chip_radius = big_chip_radius
        self.big_chip_rough = big_chip_rough
        self.second_chip_frac = second_chip_frac
        self.edge_shade = edge_shade
        self.edge_width_frac = edge_width_frac
        self.shadow_margin_frac = shadow_margin_frac

    # Builds the per-angle bite depth from the two noise bands and reads off,
    # for each pixel, whether it sits closer to the edge than its angle's bite.
    # @params rng: numpy generator
    # @params dist: distance transform of the coin interior
    # @params cy, cx: coin centroid
    # @params radius: effective coin radius
    # @params H, W: image size
    # @output signed field, positive where the rim is bitten
    def _small_chip_field(self, rng, dist, cy, cx, radius, H, W):
        N = self.n_angles
        k = np.fft.rfftfreq(N, d=1.0 / N)

        F = np.fft.rfft(rng.standard_normal(N))
        low = self.low_weight * np.exp(-(k / max(self.low_cut, 1e-6)) ** 2)
        high = self.high_weight * np.exp(
            -((k - 0.5 * (self.high_lo + self.high_hi))
              / max(0.5 * (self.high_hi - self.high_lo), 1e-6)) ** 2)
        prof = np.fft.irfft(F * (low + high), N)
        prof = (prof - prof.min()) / max(np.ptp(prof), 1e-6)
        chip = np.clip((prof - self.threshold) / max(1.0 - self.threshold, 1e-6),
                       0.0, 1.0) ** self.sharpness

        # a second, gentler profile keeps even "intact" rim off a perfect circle
        Fw = np.fft.rfft(rng.standard_normal(N))
        wav = np.fft.irfft(Fw * np.exp(-(k / 7.0) ** 2), N)
        wav = (wav - wav.min()) / max(np.ptp(wav), 1e-6)

        depth1d = radius * (self.amplitude * chip + self.waviness * wav)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        theta = np.arctan2(yy - cy, xx - cx)
        idx = (((theta + np.pi) / (2.0 * np.pi)) * N).astype(np.int64) % N
        depth_map = depth1d[idx].astype(np.float32)
        return depth_map - dist

    # Walks outward along one direction to find where the rim really is, since
    # a real flan is never a perfect circle.
    # @params coin: boolean coin mask
    # @params cy, cx: coin centroid
    # @params ang: direction in radians
    # @params r_max: effective coin radius
    # @output rim distance along that direction
    def _rim_radius(self, coin, cy, cx, ang, r_max):
        ts = np.linspace(0.3 * r_max, 1.6 * r_max, 260)
        ys = np.clip((cy + ts * np.sin(ang)).astype(int), 0, coin.shape[0] - 1)
        xs = np.clip((cx + ts * np.cos(ang)).astype(int), 0, coin.shape[1] - 1)
        hit = coin[ys, xs]
        if not hit.any():
            return r_max
        return float(ts[np.where(hit)[0][-1]])

    # Now and then takes one big circular bite out of the rim (occasionally a
    # second one roughly opposite), with a roughened edge so it isn't a perfect
    # circle.
    # @params rng: numpy generator
    # @params coin: boolean coin mask
    # @params cy, cx: coin centroid
    # @params radius: effective coin radius
    # @params H, W: image size
    # @output (signed field positive inside the clips, list of clip records)
    def _big_chip_field(self, rng, coin, cy, cx, radius, H, W):
        field = np.full((H, W), -1e6, dtype=np.float32)
        chips = []
        if self.big_chip_prob <= 0 or rng.random() >= self.big_chip_prob:
            return field, chips

        n = 2 if rng.random() < self.second_chip_frac else 1
        angles = [float(rng.uniform(0, 2 * np.pi))]
        if n == 2:
            angles.append(angles[0] + float(rng.uniform(0.8 * np.pi, 1.2 * np.pi)))

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        M = 720
        for ang in angles:
            Rc = radius * float(rng.uniform(*self.big_chip_radius))
            pen = radius * float(rng.uniform(*self.big_chip_depth))
            r_rim = self._rim_radius(coin, cy, cx, ang, radius)
            ccy = cy + (r_rim + Rc - pen) * np.sin(ang)
            ccx = cx + (r_rim + Rc - pen) * np.cos(ang)

            rough = _angular_noise(rng, M)
            r1d = Rc * (1.0 + self.big_chip_rough * rough)

            phi = np.arctan2(yy - ccy, xx - ccx)
            idx = (((phi + np.pi) / (2.0 * np.pi)) * M).astype(np.int64) % M
            d = np.hypot(yy - ccy, xx - ccx)
            field = np.maximum(field, r1d[idx] - d)
            chips.append(dict(angle=round(ang, 3), radius=round(Rc, 1),
                              depth=round(pen, 1)))
        return field, chips

    # median colour of the backdrop, so bites blend into the background
    def _background_color(self, image, coin):
        bg = ~ndimage.binary_dilation(coin, iterations=2)
        if bg.sum() < 50:
            return np.array([1.0, 1.0, 1.0], np.float32)
        return np.array([float(np.median(image[..., c][bg]))
                         for c in range(3)], np.float32)

    # Draws the dark contour along each cut: a thin line hugging the break and a
    # gradient band fading inward, both broken up by noise. Without this a chip
    # looks like an eraser stroke instead of a broken edge.
    # @params result: the image with the bites already filled
    # @params s: the signed bite field
    # @params bite: boolean map of bitten pixels
    # @params new_coin: coin mask after biting
    # @params rng: numpy generator
    # @params radius: effective coin radius
    # @output the shaded image
    def _shade_cut_edges(self, result, s, bite, new_coin, rng, radius):
        w = max(2.0, self.edge_width_frac * radius)
        noise = ndimage.gaussian_filter(
            rng.standard_normal(s.shape).astype(np.float32), 3.0)
        noise = np.clip(1.0 + 0.25 * (noise / max(noise.std(), 1e-6)), 0.7, 1.3)

        near_bite = np.clip(
            ndimage.gaussian_filter(bite.astype(np.float32), 2.0) * 3.0, 0.0, 1.0)
        line = np.exp(-(s / 1.3) ** 2) * near_bite

        band = np.zeros_like(s)
        inner = new_coin & (s > -w) & (s <= 0)
        band[inner] = (1.0 + s[inner] / w) ** 1.8

        shade = 1.0 - np.clip(0.55 * line + self.edge_shade * band, 0.0, 0.75) * noise
        return result * np.clip(shade, 0.0, 1.0)[..., None]

    # Runs the chipping on one face.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of the coin pixels
    # @params seed: seed for all the chip randomness
    # @output DamageResult with the bite map as damage_mask and the shrunken
    #         silhouette as coin_mask
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        rng = np.random.default_rng(seed)
        H, W = coin_mask.shape
        coin = coin_mask > 0.5
        area = int(coin.sum())
        if area == 0 or self.amplitude <= 0:
            return DamageResult(image.copy(), np.zeros((H, W), np.float32),
                                coin_mask, dict(filter="chip", seed=seed,
                                                amplitude=0.0))

        radius = float(np.sqrt(area / np.pi))
        ys, xs = np.where(coin)
        cy, cx = float(ys.mean()), float(xs.mean())
        dist = ndimage.distance_transform_edt(coin).astype(np.float32)

        # combine the small nibbles and the big clip into one bite field
        s_small = self._small_chip_field(rng, dist, cy, cx, radius, H, W)
        s_big, chips = self._big_chip_field(rng, coin, cy, cx, radius, H, W)
        s = np.maximum(s_small, s_big)

        alpha = np.clip((s + 0.6) / 1.2, 0.0, 1.0).astype(np.float32)
        alpha[~coin] = 0.0
        bite = alpha > 0.5
        new_coin = coin & ~bite

        # a soft halo just outside the old rim hides mask-edge artifacts
        margin = max(4.0, self.shadow_margin_frac * radius)
        d_out = ndimage.distance_transform_edt(~coin).astype(np.float32)
        halo = np.clip((s + 0.6) / 1.2, 0.0, 1.0) * np.clip(1.0 - d_out / margin, 0.0, 1.0)
        halo[coin] = 0.0
        halo = ndimage.gaussian_filter(halo.astype(np.float32), 1.0)

        # paint the bites with backdrop colour, then shade the fresh edges
        bg = self._background_color(image, coin)
        fill_a = np.maximum(alpha, halo)[..., None]
        result = image.astype(np.float32).copy()
        result = result * (1.0 - fill_a) + bg[None, None, :] * fill_a
        result = self._shade_cut_edges(result, s, bite, new_coin, rng, radius)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        new_mask = (coin_mask * (1.0 - alpha)).astype(np.float32)
        return DamageResult(result, alpha, new_mask,
                            dict(filter="chip", seed=seed,
                                 amplitude=self.amplitude,
                                 threshold=self.threshold,
                                 high_band=(self.high_lo, self.high_hi),
                                 big_chip_prob=self.big_chip_prob,
                                 big_chips=chips))
