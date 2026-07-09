from __future__ import annotations

import numpy as np
from scipy import ndimage


# Rec. 601 luminance of an RGB image.
# @params img: float RGB image
# @output float luminance map
def _lum(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# Normalizes a vector to unit length.
# @params v: vector
# @output unit vector
def _norm(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


# Recovers the photo's light direction by fitting luminance against surface
# normals derived from a band-passed height field. Falls back to upper-left,
# the numismatic photography convention, when the fit is degenerate.
# @params L: luminance map of the original image
# @params inside: boolean coin mask
# @params macro_sigma / bowl_sigma: band-pass blurs for the height field
# @params elevation: fixed z component of the light
# @output unit light direction vector
def estimate_light(L, inside, macro_sigma=8.0, bowl_sigma=40.0, elevation=0.6):
    z = ndimage.gaussian_filter(L, macro_sigma) - ndimage.gaussian_filter(L, bowl_sigma)
    zy, zx = np.gradient(z)
    nz = 1.0 / np.sqrt(zx * zx + zy * zy + 1.0)
    nx, ny = -zx * nz, -zy * nz
    magnitude = np.hypot(zx, zy)

    if not inside.any():
        return _norm([-0.5, -0.5, elevation])
    selected = inside & (magnitude > np.percentile(magnitude[inside], 55.0))
    if selected.sum() < 50:
        return _norm([-0.5, -0.5, elevation])

    A = np.stack([np.ones(int(selected.sum())), nx[selected], ny[selected]], 1)
    try:
        coef, *_ = np.linalg.lstsq(A, L[selected], rcond=None)
        lx, ly = float(coef[1]), float(coef[2])
    except Exception:
        lx, ly = -0.5, -0.5

    n = float(np.hypot(lx, ly))
    if n < 1e-4:
        return _norm([-0.5, -0.5, elevation])
    lx, ly = lx / n * 0.8, ly / n * 0.8
    return _norm([lx, ly, elevation])


# Builds surface normals from a band-passed height field, optionally mixing in
# fine detail for small-scale shading.
# @params L: luminance map of the original image
# @params macro_sigma / bowl_sigma: band-pass blurs
# @params height_scale: height exaggeration
# @params fine_sigma / fine_amt: fine-detail band and its weight
# @output (nx, ny, nz) normal component maps
def macro_normals(L, macro_sigma=8.0, bowl_sigma=40.0, height_scale=5.0,
                  fine_sigma=1.5, fine_amt=0.0):
    z = (ndimage.gaussian_filter(L, macro_sigma)
         - ndimage.gaussian_filter(L, bowl_sigma)) * height_scale
    if fine_amt > 0:
        z = z + fine_amt * (L - ndimage.gaussian_filter(L, fine_sigma)) * height_scale
    zy, zx = np.gradient(z)
    nz = 1.0 / np.sqrt(zx * zx + zy * zy + 1.0)
    return (-zx * nz).astype(np.float32), (-zy * nz).astype(np.float32), nz.astype(np.float32)


# Samples the coin's own polished-highlight color from its brightest pixels.
# @params rgb: float RGB image
# @params inside: boolean coin mask
# @params bright_pct: luminance percentile that counts as highlight
# @output RGB tone vector
def silver_tone(rgb, inside, bright_pct=80.0):
    L = _lum(rgb)
    if not inside.any():
        return np.array([0.75, 0.74, 0.70], np.float32)
    threshold = np.percentile(L[inside], bright_pct)
    bright = inside & (L >= threshold)
    if bright.sum() < 20:
        bright = inside
    return np.array([float(np.median(rgb[..., c][bright])) for c in range(3)], np.float32)


# Blinn-Phong specular layer: surfaces facing the light glint, gated away from
# the coin's dark recesses, with optional hairline scratches carved out so the
# shine is interrupted the way worn silver really is.
# @params nx, ny, nz: normal maps
# @params light: light direction
# @params L_for_gate: luminance used to gate recessed areas
# @params inside: boolean coin mask
# @params shininess: specular exponent
# @params gate_pct: luminance percentile below which no shine appears
# @params broaden: final blur of the layer
# @params grain: hairline scratch strength
# @params rng: generator for the scratch directions
# @output float specular map
def specular_layer(nx, ny, nz, light, L_for_gate, inside,
                   shininess=12.0, gate_pct=45.0, broaden=1.2,
                   grain=0.0, rng=None):
    l = _norm(light)
    half = _norm(l + np.array([0.0, 0.0, 1.0], np.float32))
    spec = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0.0, 1.0) ** shininess

    macro = ndimage.gaussian_filter(L_for_gate, 8.0)
    if inside.any():
        lo = np.percentile(macro[inside], gate_pct)
        hi = np.percentile(macro[inside], 99.0)
        gate = np.clip((macro - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        gate = macro

    out = ndimage.gaussian_filter(spec * gate, broaden)

    if grain > 0.0 and rng is not None:
        H, W = out.shape
        texture = np.zeros((H, W), np.float32)
        base_angle = float(rng.uniform(0.0, 180.0))
        for delta, weight in ((0.0, 1.0), (37.0, 0.5)):
            noise = ndimage.gaussian_filter(
                rng.standard_normal((H, W)).astype(np.float32), (0.5, 4.5))
            noise = ndimage.rotate(noise, base_angle + delta, reshape=False,
                                   order=1, mode="reflect")
            noise /= max(noise.std(), 1e-6)
            texture += weight * np.clip(noise - 1.05, 0.0, None)
        out = out * np.clip(1.0 - grain * texture, 0.0, 1.0)

    return out.astype(np.float32)


# Finds large dark blobs the wear process created and lifts them toward the
# local average: heavy wear should flatten contrast, not deposit black smudges.
# @params worn: float RGB image after wear
# @params orig_mean: mean luminance of the original, to keep brightness stable
# @params inside: boolean coin mask
# @params lift: how strongly the blobs are brightened
# @params field_sigma: blur radius of the local average
# @params dark_thr: darkness level that counts as a blob
# @params min_area: smallest blob to fix, in pixels
# @params rim_erode: pixels eroded from the rim before searching
# @params keep_mean: rescale luminance back to the original mean
# @output corrected image
def reduce_big_shadows(worn, orig_mean, inside, lift=0.6, field_sigma=16.0,
                       dark_thr=0.36, min_area=90, rim_erode=8, keep_mean=True):
    Lw = _lum(worn)
    field = ndimage.gaussian_filter(Lw, field_sigma)
    darkness = np.clip(field - Lw, 0.0, None)
    if not inside.any():
        return worn

    hi = np.percentile(darkness[inside], 95.0)
    dark_norm = np.clip(darkness / max(hi, 1e-6), 0.0, 1.0)
    core = ndimage.binary_erosion(inside, iterations=rim_erode)
    blobs = (dark_norm > dark_thr) & core

    labels, count = ndimage.label(blobs)
    if count:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
        blobs = np.isin(labels, [i + 1 for i in range(count) if sizes[i] >= min_area])

    weight = ndimage.gaussian_filter(blobs.astype(np.float32), 5.0)
    weight = weight / max(weight.max(), 1e-6) * dark_norm
    lifted = Lw + lift * weight * (field - Lw)

    if keep_mean:
        lifted_mean = float(lifted[inside].mean())
        if lifted_mean > 1e-6:
            lifted = lifted * (orig_mean / lifted_mean)

    ratio = np.clip(lifted / np.clip(Lw, 0.04, None), 0.0, 3.0)[..., None]
    return np.clip(worn * ratio, 0.0, 1.0).astype(np.float32)


# Re-lights a sanded coin so it reads as polished silver: pulls the color
# toward the coin's own metal tone, adds a specular glint from the estimated
# light direction, and renormalizes brightness.
# @params B: the sanded image
# @params original: the original image, used for normals and light estimation
# @params inside: boolean coin mask
# @params light: light direction, estimated from the photo when None
# @params silver_boost: strength of the pull toward the metal tone
# @params spec_gain / shininess / spec_color / gate_pct: specular settings
# @params fine_amt: fine detail in the normals
# @params brightness_bias: target brightness relative to the original
# @params silver_hue: fallback hue for nearly-gray coins
# @params spec_grain: hairline scratch strength in the specular layer
# @params rng: generator for the scratches
# @output (relit image, light direction used, specular map)
def silverize_and_shine(B, original, inside, light=None,
                        silver_boost=0.35, spec_gain=0.5, shininess=12.0,
                        spec_color=(1.0, 0.99, 0.95), gate_pct=45.0,
                        fine_amt=0.25, brightness_bias=1.0,
                        silver_hue=(1.03, 1.00, 0.93),
                        spec_grain=0.0, rng=None):
    B = B.astype(np.float32)
    L_orig = _lum(original.astype(np.float32))
    if light is None:
        light = estimate_light(L_orig, inside)

    tone = silver_tone(original.astype(np.float32), inside)
    tone_max, tone_min = float(tone.max()), float(tone.min())
    saturation = (tone_max - tone_min) / max(tone_max, 1e-6)
    if saturation > 0.04:
        hue = tone / max(float(tone.mean()), 1e-6)
    else:
        hue = np.asarray(silver_hue, np.float32)
    hue = hue / max(float(hue.mean()), 1e-6)

    Lb = _lum(B)
    silver = np.clip(Lb[..., None] * hue[None, None, :], 0.0, 1.0)
    out = (1.0 - silver_boost) * B + silver_boost * silver

    nx, ny, nz = macro_normals(L_orig, fine_amt=fine_amt)
    spec = specular_layer(nx, ny, nz, light, L_orig, inside,
                          shininess=shininess, gate_pct=gate_pct,
                          grain=spec_grain, rng=rng)
    spec_rgb = np.asarray(spec_color, np.float32)[None, None, :]
    out = out + spec_gain * spec[..., None] * spec_rgb * (1.0 - out)

    if inside.any():
        target = float(L_orig[inside].mean()) * brightness_bias
        current = float(_lum(out)[inside].mean())
        if current > 1e-6:
            scale = min(target / current, 1.0 + 0.10 * brightness_bias)
            out = out * scale

    return np.clip(out, 0.0, 1.0).astype(np.float32), _norm(light), spec
