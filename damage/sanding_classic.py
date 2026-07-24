import numpy as np
from scipy import ndimage

from .base import DamageFilter, DamageResult
from .height import SignedRelief


# per-pixel blend: t=0 gives a, t=1 gives b
def _lerp(a, b, t):
    return (1.0 - t[..., None]) * a + t[..., None] * b


# cubic ease from 0 to 1 across [lo, hi], flat outside
def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# boolean disk footprint of the given radius, for morphology
def _disk(radius):
    r = int(max(round(radius), 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


# Opening then closing with a disk of radius `size` pixels. This deletes any
# bright or dark feature smaller than the disk outright, which is what abrasion
# does to fine relief (a plain blur would only dim it and leave a smudge).
def _morph_smooth(img, size):
    if size < 1:
        return img
    fp = _disk(size)
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        opened = ndimage.grey_opening(img[..., c], footprint=fp)
        out[..., c] = ndimage.grey_closing(opened, footprint=fp)
    return out


# Perona-Malik anisotropic diffusion. Detail melts away inside regions but
# strong edges hold, so a worn design still reads by its outline. Each of the
# n_iter passes lets intensity flow to the four neighbours, weighted so a jump
# bigger than kappa (an edge) almost blocks the flow. step is the integration
# step size.
def _perona_malik(img, n_iter, kappa, step=0.20):
    if n_iter < 1:
        return img.astype(np.float32)

    out = img.astype(np.float32).copy()
    for _ in range(int(n_iter)):
        for c in range(out.shape[-1]):
            I = out[..., c]

            # differences to the four neighbours
            dn = np.zeros_like(I); dn[1:, :] = I[:-1, :] - I[1:, :]
            ds = np.zeros_like(I); ds[:-1, :] = I[1:, :] - I[:-1, :]
            de = np.zeros_like(I); de[:, :-1] = I[:, 1:] - I[:, :-1]
            dw = np.zeros_like(I); dw[:, 1:] = I[:, :-1] - I[:, 1:]

            # a jump bigger than kappa is an edge, so damp the flow across it
            cn = np.exp(-(dn / kappa) ** 2)
            cs = np.exp(-(ds / kappa) ** 2)
            ce = np.exp(-(de / kappa) ** 2)
            cw = np.exp(-(dw / kappa) ** 2)

            out[..., c] = I + step * (cn * dn + cs * ds + ce * de + cw * dw)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Split an image into coarse and fine detail with a Gaussian low-pass in the
# frequency domain; cutoff is in cycles per pixel. low + high adds back up to
# the original, and that's what we return.
def _fft_bands(img, cutoff):
    H, W = img.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    lowpass = np.exp(-(radius / max(cutoff, 1e-6)) ** 2)

    low = np.empty_like(img)
    for c in range(img.shape[-1]):
        F = np.fft.rfft2(img[..., c])
        low[..., c] = np.fft.irfft2(F * lowpass, s=(H, W))
    high = img - low
    return low.astype(np.float32), high.astype(np.float32)


# noise smeared along one direction: the seed of a scratch streak
def _directional_streak(H, W, rng, angle, length):
    n = rng.standard_normal((H, W)).astype(np.float32)
    streak = ndimage.gaussian_filter(n, (0.6, length))
    if abs(angle) > 1e-3:
        streak = ndimage.rotate(streak, np.degrees(angle), reshape=False,
                                order=1, mode="reflect")
    return (streak / max(streak.std(), 1e-6)).astype(np.float32)


# Two streak directions plus a bit of fine grain, mixed into the faint scratch
# signature that circulation leaves on the surface. angle is the main scratch
# direction in radians, scratch_len the streak length in pixels, and
# scratch_weight trades the streaks off against isotropic grain. The result is
# zero-mean with unit spread.
def _abrasion_texture(H, W, rng, angle, scratch_len=3.5, scratch_weight=0.5):
    s1 = _directional_streak(H, W, rng, angle, scratch_len)
    s2 = _directional_streak(H, W, rng, angle + np.pi / 3.0, scratch_len)
    streak = 0.6 * s1 + 0.4 * s2
    streak /= max(streak.std(), 1e-6)

    grain = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.6)
    grain /= max(grain.std(), 1e-6)

    tex = scratch_weight * streak + (1.0 - scratch_weight) * grain
    tex -= float(tex.mean())
    tex /= max(tex.std(), 1e-6)
    return tex.astype(np.float32)


# The wear stage of the simulator. It works out which parts of the coin stand
# proud, presses an imaginary (slightly wavy, rim-dipping) sanding plane onto
# them, cooks up a "worn metal" fill, and crossfades to it by the wear mask.
class ClassicSandingFilter(DamageFilter):

    def __init__(self,
                 height_estimator=None,
                 depth=0.35,
                 softness=0.07,
                 spread=0.0,
                 knockdown_size=4.0,
                 diffusion_iter=8,
                 diffusion_kappa=0.05,
                 fft_cutoff=0.12,
                 hf_attenuation=0.80,
                 lowfreq_keep=1.0,
                 plane_jitter=0.06,
                 noise_sigma=8.0,
                 radial_bias=0.15,
                 burnish_amount=0.45,
                 abrasion_texture=0.06):
        # relief estimator, defaulting to SignedRelief
        self.height_estimator = height_estimator or SignedRelief()

        # where the imaginary sanding plane bites
        self.depth = depth                    # how far down the relief it reaches
        self.softness = softness              # half-width of the soft contact band
        self.spread = spread                  # dilation that merges nearby worn patches
        self.knockdown_size = knockdown_size  # smallest feature that survives, in pixels

        # how the worn-metal fill is cooked up: diffusion then a frequency mix
        self.diffusion_iter = diffusion_iter
        self.diffusion_kappa = diffusion_kappa
        self.fft_cutoff = fft_cutoff
        self.hf_attenuation = hf_attenuation
        self.lowfreq_keep = lowfreq_keep

        # the plane isn't flat: it wobbles and sags toward the rim
        self.plane_jitter = plane_jitter      # strength of the wobble
        self.noise_sigma = noise_sigma        # blur radius of that wobble
        self.radial_bias = radial_bias        # how hard it sags toward the rim

        self.burnish_amount = burnish_amount  # pull of the fill toward the highlight tone
        self.abrasion_texture = abrasion_texture   # strength of the scratch texture

    # every knob moves the same way from Very Fine down to About Good
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

    # build the filter from a named grade preset
    @classmethod
    def for_grade(cls, grade, height_estimator=None):
        return cls(height_estimator=height_estimator or SignedRelief(),
                   **cls._GRADES[grade])

    # Run the wear on one face. seed drives the plane jitter and the scratch
    # randomness. Returns a DamageResult with the wear mask as its damage_mask.
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        rng = np.random.default_rng(seed)
        img = image.astype(np.float32)
        H, W = coin_mask.shape
        cm = coin_mask[..., None]
        inside = coin_mask > 0.5

        # start from the relief estimate, then bend the "plane" two ways
        height = self.height_estimator.estimate(image, coin_mask)

        if self.plane_jitter > 0:
            jitter = rng.standard_normal((H, W)).astype(np.float32)
            jitter = ndimage.gaussian_filter(jitter, self.noise_sigma)
            jitter /= max(jitter.std(), 1e-6)
            height = height + self.plane_jitter * jitter

        if self.radial_bias > 0 and inside.any():
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            ys, xs = np.where(inside)
            cy, cx = float(ys.mean()), float(xs.mean())
            coin_radius = max(float(np.sqrt(inside.sum() / np.pi)), 1e-6)
            r_norm = np.clip(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / coin_radius,
                             0.0, 1.5)
            height = height + self.radial_bias * (r_norm - 0.5)

        # where does the plane bite? soft contact, then snap away the faint fog
        plane = 1.0 - self.depth
        wear = _smoothstep(plane - self.softness, plane + self.softness, height)
        wear = _smoothstep(0.15, 0.85, wear)
        if self.spread >= 1:
            wear = ndimage.grey_dilation(wear, footprint=_disk(self.spread))
        wear = (wear * coin_mask).astype(np.float32)

        # manufacture the worn-metal surface: diffuse, knock down, kill detail
        fill = _perona_malik(img, self.diffusion_iter, self.diffusion_kappa)
        fill = _morph_smooth(fill, self.knockdown_size)
        low, high = _fft_bands(fill, self.fft_cutoff)
        fill = np.clip(low * self.lowfreq_keep + high * (1.0 - self.hf_attenuation),
                       0.0, 1.0).astype(np.float32)

        # rubbed high points polish bright, toward the coin's own highlight tone
        if self.burnish_amount > 0 and inside.any():
            lum = (0.299 * image[..., 0] + 0.587 * image[..., 1]
                   + 0.114 * image[..., 2])
            thr = np.percentile(lum[inside], 85.0)
            bright = inside & (lum >= thr)
            if bright.any():
                tone = np.array([float(np.median(image[..., c][bright]))
                                 for c in range(3)], dtype=np.float32)
                tone_map = np.ones_like(fill) * tone
                amt = np.full((H, W), self.burnish_amount, dtype=np.float32)
                fill = _lerp(fill, tone_map, amt)

        if self.abrasion_texture > 0:
            angle = float(rng.uniform(0, np.pi))
            tex = _abrasion_texture(H, W, rng, angle)
            fill = fill + self.abrasion_texture * tex[..., None]

        # crossfade original -> fill by the wear mask, keep the background as-is
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
