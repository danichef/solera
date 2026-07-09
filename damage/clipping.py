from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import DamageFilter, DamageResult


# Band-limited noise around a circle, used to roughen chip outlines.
# @params rng: numpy random generator
# @params n: number of angular samples
# @params low, high: frequency band in cycles per revolution
# @output unit-spread profile of length n
def _angular_noise(rng, n, low=2.0, high=9.0):
    k = np.fft.rfftfreq(n, d=1.0 / n)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    band = np.exp(-((k - 0.5 * (low + high)) / max(0.5 * (high - low), 1e-6)) ** 2)
    band[0] = 0.0
    profile = np.fft.irfft(spectrum * band, n)
    return (profile / max(profile.std(), 1e-6)).astype(np.float32)


# Bites chips out of the coin's silhouette: small nibbles along the rim from a
# thresholded ring of noise, plus an occasional large circular flan clip. The
# cut edges get a dark contour so they read as broken metal, and the coin mask
# is shrunk to the new outline.
class ChipFilter(DamageFilter):

    # @params amplitude: depth of the rim nibbles, relative to the coin radius
    # @params threshold: noise level a peak must exceed to become a chip
    # @params low_weight / high_weight / low_cut / high_lo / high_hi: the two
    #         frequency bands of the rim noise ring
    # @params sharpness: exponent shaping the bite profile
    # @params waviness: subtle irregularity applied to the whole rim
    # @params n_angles: angular resolution of the rim profile
    # @params big_chip_prob: chance of a large flan clip
    # @params big_chip_depth / big_chip_radius: clip size ranges, in radii
    # @params big_chip_rough: roughness of the clip outline
    # @params second_chip_frac: chance a clip gets a partner on the far side
    # @params edge_shade: darkness of the shading band along the cut
    # @params edge_width_frac / shadow_margin_frac: geometry of that shading
    def __init__(self,
                 amplitude: float = 0.07,
                 threshold: float = 0.48,
                 low_weight: float = 1.0,
                 high_weight: float = 0.32,
                 low_cut: float = 3.5,
                 high_lo: float = 9.0,
                 high_hi: float = 20.0,
                 sharpness: float = 0.7,
                 waviness: float = 0.010,
                 n_angles: int = 1440,
                 big_chip_prob: float = 0.0,
                 big_chip_depth: tuple = (0.06, 0.15),
                 big_chip_radius: tuple = (0.45, 0.95),
                 big_chip_rough: float = 0.03,
                 second_chip_frac: float = 0.30,
                 edge_shade: float = 0.55,
                 edge_width_frac: float = 0.030,
                 shadow_margin_frac: float = 0.045):
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

    # Builds the per-angle bite depth from two noise bands and compares it to
    # each pixel's distance from the rim.
    # @params rng: numpy random generator
    # @params dist: distance transform of the coin interior
    # @params cy, cx: coin centroid
    # @params radius: effective coin radius
    # @params H, W: image size
    # @output signed field, positive where the rim is bitten
    def _small_chip_field(self, rng, dist, cy, cx, radius, H, W):
        N = self.n_angles
        k = np.fft.rfftfreq(N, d=1.0 / N)

        spectrum = np.fft.rfft(rng.standard_normal(N))
        low = self.low_weight * np.exp(-(k / max(self.low_cut, 1e-6)) ** 2)
        high = self.high_weight * np.exp(
            -((k - 0.5 * (self.high_lo + self.high_hi))
              / max(0.5 * (self.high_hi - self.high_lo), 1e-6)) ** 2)
        profile = np.fft.irfft(spectrum * (low + high), N)
        profile = (profile - profile.min()) / max(np.ptp(profile), 1e-6)
        chip = np.clip((profile - self.threshold) / max(1.0 - self.threshold, 1e-6),
                       0.0, 1.0) ** self.sharpness

        wave_spectrum = np.fft.rfft(rng.standard_normal(N))
        wave = np.fft.irfft(wave_spectrum * np.exp(-(k / 7.0) ** 2), N)
        wave = (wave - wave.min()) / max(np.ptp(wave), 1e-6)

        depth_per_angle = radius * (self.amplitude * chip + self.waviness * wave)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        theta = np.arctan2(yy - cy, xx - cx)
        index = (((theta + np.pi) / (2.0 * np.pi)) * N).astype(np.int64) % N
        depth_map = depth_per_angle[index].astype(np.float32)
        return depth_map - dist

    # Walks outward along one direction to find where the rim actually is,
    # since real flans are not perfect circles.
    # @params coin: boolean coin mask
    # @params cy, cx: coin centroid
    # @params angle: direction in radians
    # @params r_max: effective coin radius
    # @output rim distance along that direction
    def _rim_radius(self, coin, cy, cx, angle, r_max):
        steps = np.linspace(0.3 * r_max, 1.6 * r_max, 260)
        ys = np.clip((cy + steps * np.sin(angle)).astype(int), 0, coin.shape[0] - 1)
        xs = np.clip((cx + steps * np.cos(angle)).astype(int), 0, coin.shape[1] - 1)
        hits = coin[ys, xs]
        if not hits.any():
            return r_max
        return float(steps[np.where(hits)[0][-1]])

    # Occasionally takes one (or two opposite) large circular bites out of the
    # rim, with a roughened outline so the clip is not geometrically perfect.
    # @params rng: numpy random generator
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

        count = 2 if rng.random() < self.second_chip_frac else 1
        angles = [float(rng.uniform(0, 2 * np.pi))]
        if count == 2:
            angles.append(angles[0] + float(rng.uniform(0.8 * np.pi, 1.2 * np.pi)))

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        M = 720
        for angle in angles:
            clip_radius = radius * float(rng.uniform(*self.big_chip_radius))
            penetration = radius * float(rng.uniform(*self.big_chip_depth))
            rim = self._rim_radius(coin, cy, cx, angle, radius)
            center_y = cy + (rim + clip_radius - penetration) * np.sin(angle)
            center_x = cx + (rim + clip_radius - penetration) * np.cos(angle)

            rough = _angular_noise(rng, M)
            radius_per_angle = clip_radius * (1.0 + self.big_chip_rough * rough)

            phi = np.arctan2(yy - center_y, xx - center_x)
            index = (((phi + np.pi) / (2.0 * np.pi)) * M).astype(np.int64) % M
            distance = np.hypot(yy - center_y, xx - center_x)
            field = np.maximum(field, radius_per_angle[index] - distance)
            chips.append(dict(angle=round(angle, 3), radius=round(clip_radius, 1),
                              depth=round(penetration, 1)))
        return field, chips

    # Samples the backdrop color so bitten areas blend into the background.
    # @params image: float RGB image
    # @params coin: boolean coin mask
    # @output RGB background color
    def _background_color(self, image, coin):
        background = ~ndimage.binary_dilation(coin, iterations=2)
        if background.sum() < 50:
            return np.array([1.0, 1.0, 1.0], np.float32)
        return np.array([float(np.median(image[..., c][background]))
                         for c in range(3)], np.float32)

    # Draws the dark contour along each cut: a thin line hugging the break plus
    # a gradient band fading inward, both modulated by noise. Without this a
    # chip reads as an eraser mark instead of broken metal.
    # @params result: image with the bites already filled
    # @params s: the signed bite field
    # @params bite: boolean map of bitten pixels
    # @params new_coin: boolean mask of the coin after biting
    # @params rng: numpy random generator
    # @params radius: effective coin radius
    # @output shaded image
    def _shade_cut_edges(self, result, s, bite, new_coin, rng, radius):
        width = max(2.0, self.edge_width_frac * radius)
        noise = ndimage.gaussian_filter(
            rng.standard_normal(s.shape).astype(np.float32), 3.0)
        noise = np.clip(1.0 + 0.25 * (noise / max(noise.std(), 1e-6)), 0.7, 1.3)

        near_bite = np.clip(
            ndimage.gaussian_filter(bite.astype(np.float32), 2.0) * 3.0, 0.0, 1.0)
        line = np.exp(-(s / 1.3) ** 2) * near_bite

        band = np.zeros_like(s)
        inner = new_coin & (s > -width) & (s <= 0)
        band[inner] = (1.0 + s[inner] / width) ** 1.8

        shade = 1.0 - np.clip(0.55 * line + self.edge_shade * band, 0.0, 0.75) * noise
        return result * np.clip(shade, 0.0, 1.0)[..., None]

    # Applies the chipping to one face.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of coin pixels
    # @params seed: seed for all chip randomness
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

        small_field = self._small_chip_field(rng, dist, cy, cx, radius, H, W)
        big_field, chips = self._big_chip_field(rng, coin, cy, cx, radius, H, W)
        s = np.maximum(small_field, big_field)

        alpha = np.clip((s + 0.6) / 1.2, 0.0, 1.0).astype(np.float32)
        alpha[~coin] = 0.0
        bite = alpha > 0.5
        new_coin = coin & ~bite

        margin = max(4.0, self.shadow_margin_frac * radius)
        dist_outside = ndimage.distance_transform_edt(~coin).astype(np.float32)
        halo = np.clip((s + 0.6) / 1.2, 0.0, 1.0)
        halo = halo * np.clip(1.0 - dist_outside / margin, 0.0, 1.0)
        halo[coin] = 0.0
        halo = ndimage.gaussian_filter(halo.astype(np.float32), 1.0)

        background = self._background_color(image, coin)
        fill_alpha = np.maximum(alpha, halo)[..., None]
        result = image.astype(np.float32).copy()
        result = result * (1.0 - fill_alpha) + background[None, None, :] * fill_alpha
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
