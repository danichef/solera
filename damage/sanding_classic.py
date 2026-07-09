from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import DamageFilter, DamageResult
from .height import SignedRelief


# Linear blend between two images with a per-pixel weight.
# @params a, b: float RGB images
# @params t: float weight map, 0 = a, 1 = b
# @output blended image
def _lerp(a, b, t):
    return (1.0 - t[..., None]) * a + t[..., None] * b


# Cubic ease from 0 to 1 across the [lo, hi] band, flat outside it.
# @params lo, hi: band edges
# @params x: input array
# @output eased array in [0, 1]
def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# Builds a disk-shaped boolean footprint for morphology.
# @params radius: disk radius in pixels
# @output boolean array
def _disk(radius: float) -> np.ndarray:
    r = int(max(round(radius), 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


# Grey opening then closing with a disk: physically removes any bright or
# dark feature smaller than the disk, the way abrasion planes off small relief.
# @params img: float RGB image
# @params size: disk radius in pixels
# @output smoothed image
def _morph_smooth(img: np.ndarray, size: float) -> np.ndarray:
    if size < 1:
        return img
    footprint = _disk(size)
    out = np.empty_like(img)
    for channel in range(img.shape[-1]):
        opened = ndimage.grey_opening(img[..., channel], footprint=footprint)
        out[..., channel] = ndimage.grey_closing(opened, footprint=footprint)
    return out


# Perona-Malik anisotropic diffusion: detail melts within regions while strong
# edges survive, so a worn design stays attributable by its outline.
# @params img: float RGB image
# @params n_iter: diffusion iterations
# @params kappa: edge threshold; differences above it block the flow
# @params step: integration step size
# @output diffused image
def _perona_malik(img: np.ndarray, n_iter: int, kappa: float,
                  step: float = 0.20) -> np.ndarray:
    if n_iter < 1:
        return img.astype(np.float32)

    out = img.astype(np.float32).copy()
    for _ in range(int(n_iter)):
        for channel in range(out.shape[-1]):
            I = out[..., channel]
            dn = np.zeros_like(I); dn[1:, :] = I[:-1, :] - I[1:, :]
            ds = np.zeros_like(I); ds[:-1, :] = I[1:, :] - I[:-1, :]
            de = np.zeros_like(I); de[:, :-1] = I[:, 1:] - I[:, :-1]
            dw = np.zeros_like(I); dw[:, 1:] = I[:, :-1] - I[:, 1:]
            cn = np.exp(-(dn / kappa) ** 2)
            cs = np.exp(-(ds / kappa) ** 2)
            ce = np.exp(-(de / kappa) ** 2)
            cw = np.exp(-(dw / kappa) ** 2)
            out[..., channel] = I + step * (cn * dn + cs * ds + ce * de + cw * dw)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Splits an image into low and high spatial frequencies with a Gaussian
# low-pass in the Fourier domain.
# @params img: float RGB image
# @params cutoff: low-pass cutoff in cycles per pixel
# @output (low band, high band), summing back to the input
def _fft_bands(img: np.ndarray, cutoff: float):
    H, W = img.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    lowpass = np.exp(-(radius / max(cutoff, 1e-6)) ** 2)

    low = np.empty_like(img)
    for channel in range(img.shape[-1]):
        spectrum = np.fft.rfft2(img[..., channel])
        low[..., channel] = np.fft.irfft2(spectrum * lowpass, s=(H, W))
    high = img - low
    return low.astype(np.float32), high.astype(np.float32)


# Noise stretched along one direction: the raw material for scratch streaks.
# @params H, W: output size
# @params rng: numpy random generator
# @params angle: streak direction in radians
# @params length: streak length in pixels
# @output unit-spread float array
def _directional_streak(H: int, W: int, rng, angle: float, length: float) -> np.ndarray:
    noise = rng.standard_normal((H, W)).astype(np.float32)
    streak = ndimage.gaussian_filter(noise, (0.6, length))
    if abs(angle) > 1e-3:
        streak = ndimage.rotate(streak, np.degrees(angle), reshape=False,
                                order=1, mode="reflect")
    return (streak / max(streak.std(), 1e-6)).astype(np.float32)


# Combines two streak directions with fine grain into the faint scratch
# signature of circulation wear.
# @params H, W: output size
# @params rng: numpy random generator
# @params angle: main scratch direction in radians
# @params scratch_len: streak length in pixels
# @params scratch_weight: streaks vs isotropic grain mix
# @output zero-mean unit-spread texture
def _abrasion_texture(H: int, W: int, rng, angle: float,
                      scratch_len: float = 3.5, scratch_weight: float = 0.5) -> np.ndarray:
    first = _directional_streak(H, W, rng, angle, scratch_len)
    second = _directional_streak(H, W, rng, angle + np.pi / 3.0, scratch_len)
    streak = 0.6 * first + 0.4 * second
    streak /= max(streak.std(), 1e-6)

    grain = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.6)
    grain /= max(grain.std(), 1e-6)

    texture = scratch_weight * streak + (1.0 - scratch_weight) * grain
    texture -= float(texture.mean())
    texture /= max(texture.std(), 1e-6)
    return texture.astype(np.float32)


# Simulates circulation wear: estimates which parts of the coin are raised,
# decides where an imaginary sanding plane touches them, builds a "worn metal"
# fill image, and blends the two by the wear mask.
class ClassicSandingFilter(DamageFilter):

    # @params height_estimator: relief estimator, SignedRelief by default
    # @params depth: how far down the relief the sanding plane reaches
    # @params softness: half-width of the soft contact band around the plane
    # @params spread: dilation radius that merges nearby worn patches
    # @params knockdown_size: smallest feature that survives, in pixels
    # @params diffusion_iter / diffusion_kappa: Perona-Malik settings
    # @params fft_cutoff / hf_attenuation / lowfreq_keep: frequency-band mix
    # @params plane_jitter: amplitude of the wavy plane perturbation
    # @params noise_sigma: blur radius of the jitter noise
    # @params radial_bias: how strongly the plane dips toward the rim
    # @params burnish_amount: pull of the fill toward the coin's highlight tone
    # @params abrasion_texture: amplitude of the scratch texture
    def __init__(self,
                 height_estimator=None,
                 depth: float = 0.35,
                 softness: float = 0.07,
                 spread: float = 0.0,
                 knockdown_size: float = 4.0,
                 diffusion_iter: int = 8,
                 diffusion_kappa: float = 0.05,
                 fft_cutoff: float = 0.12,
                 hf_attenuation: float = 0.80,
                 lowfreq_keep: float = 1.0,
                 plane_jitter: float = 0.06,
                 noise_sigma: float = 8.0,
                 radial_bias: float = 0.15,
                 burnish_amount: float = 0.45,
                 abrasion_texture: float = 0.06):
        self.height_estimator = height_estimator or SignedRelief()
        self.depth = depth
        self.softness = softness
        self.spread = spread
        self.knockdown_size = knockdown_size
        self.diffusion_iter = diffusion_iter
        self.diffusion_kappa = diffusion_kappa
        self.fft_cutoff = fft_cutoff
        self.hf_attenuation = hf_attenuation
        self.lowfreq_keep = lowfreq_keep
        self.plane_jitter = plane_jitter
        self.noise_sigma = noise_sigma
        self.radial_bias = radial_bias
        self.burnish_amount = burnish_amount
        self.abrasion_texture = abrasion_texture

    _GRADES = {
        "vf": dict(depth=0.24, softness=0.06, spread=0, knockdown_size=3, diffusion_iter=5,
                   hf_attenuation=0.60, burnish_amount=0.18, radial_bias=0.03, abrasion_texture=0.03),
        "f":  dict(depth=0.36, softness=0.07, spread=0, knockdown_size=4, diffusion_iter=6,
                   hf_attenuation=0.70, burnish_amount=0.22, radial_bias=0.05, abrasion_texture=0.035),
        "vg": dict(depth=0.46, softness=0.07, spread=1, knockdown_size=4, diffusion_iter=8,
                   hf_attenuation=0.80, burnish_amount=0.26, radial_bias=0.08, abrasion_texture=0.04),
        "g":  dict(depth=0.56, softness=0.08, spread=1, knockdown_size=5, diffusion_iter=9,
                   hf_attenuation=0.87, burnish_amount=0.30, radial_bias=0.11, abrasion_texture=0.045),
        "ag": dict(depth=0.58, softness=0.10, spread=2, knockdown_size=6, diffusion_iter=11,
                   hf_attenuation=0.90, burnish_amount=0.34, radial_bias=0.14, abrasion_texture=0.05),
    }

    # Builds the filter from a named grade preset.
    # @params grade: one of vf / f / vg / g / ag
    # @params height_estimator: optional estimator override
    # @output configured ClassicSandingFilter
    @classmethod
    def for_grade(cls, grade: str, height_estimator=None):
        return cls(height_estimator=height_estimator or SignedRelief(),
                   **cls._GRADES[grade])

    # Applies the wear simulation to one face.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of coin pixels
    # @params seed: seed for the jitter and texture randomness
    # @output DamageResult with the wear mask as damage_mask
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        rng = np.random.default_rng(seed)
        img = image.astype(np.float32)
        H, W = coin_mask.shape
        cm = coin_mask[..., None]
        inside = coin_mask > 0.5

        height = self.height_estimator.estimate(image, coin_mask)

        if self.plane_jitter > 0:
            jitter = rng.standard_normal((H, W)).astype(np.float32)
            jitter = ndimage.gaussian_filter(jitter, self.noise_sigma)
            jitter /= max(jitter.std(), 1e-6)
            height = height + self.plane_jitter * jitter

        if self.radial_bias > 0 and inside.any():
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            ys_all, xs_all = np.where(inside)
            cy, cx = float(ys_all.mean()), float(xs_all.mean())
            coin_radius = max(float(np.sqrt(inside.sum() / np.pi)), 1e-6)
            r_norm = np.clip(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / coin_radius,
                             0.0, 1.5)
            height = height + self.radial_bias * (r_norm - 0.5)

        plane = 1.0 - self.depth
        wear = _smoothstep(plane - self.softness, plane + self.softness, height)
        wear = _smoothstep(0.15, 0.85, wear)
        if self.spread >= 1:
            wear = ndimage.grey_dilation(wear, footprint=_disk(self.spread))
        wear = (wear * coin_mask).astype(np.float32)

        fill = _perona_malik(img, self.diffusion_iter, self.diffusion_kappa)
        fill = _morph_smooth(fill, self.knockdown_size)
        low, high = _fft_bands(fill, self.fft_cutoff)
        fill = np.clip(low * self.lowfreq_keep + high * (1.0 - self.hf_attenuation),
                       0.0, 1.0).astype(np.float32)

        if self.burnish_amount > 0 and inside.any():
            lum = (0.299 * image[..., 0] + 0.587 * image[..., 1]
                   + 0.114 * image[..., 2])
            threshold = np.percentile(lum[inside], 85.0)
            bright = inside & (lum >= threshold)
            if bright.any():
                tone = np.array([float(np.median(image[..., c][bright]))
                                 for c in range(3)], dtype=np.float32)
                tone_map = np.ones_like(fill) * tone
                burnish = np.full((H, W), self.burnish_amount, dtype=np.float32)
                fill = _lerp(fill, tone_map, burnish)

        if self.abrasion_texture > 0:
            angle = float(rng.uniform(0, np.pi))
            texture = _abrasion_texture(H, W, rng, angle)
            fill = fill + self.abrasion_texture * texture[..., None]

        worn = _lerp(img, np.clip(fill, 0.0, 1.0).astype(np.float32), wear)
        img = np.clip(worn * cm + image * (1.0 - cm), 0.0, 1.0).astype(np.float32)

        params = dict(filter="sanding", seed=seed, depth=self.depth,
                      softness=self.softness, knockdown_size=self.knockdown_size,
                      diffusion_iter=self.diffusion_iter,
                      hf_attenuation=self.hf_attenuation,
                      burnish_amount=self.burnish_amount,
                      radial_bias=self.radial_bias,
                      height=type(self.height_estimator).__name__)
        return DamageResult(img, wear, coin_mask, params)
