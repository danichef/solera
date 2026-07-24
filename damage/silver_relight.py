import numpy as np
from scipy import ndimage


# Rec. 601 luminance
def _lum(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# unit-length version of a vector
def _norm(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


# Guess the lamp direction from the photo by least-squares fitting luminance
# against normals off a band-passed height field (macro_sigma and bowl_sigma
# are the two blurs, elevation fixes the light's z). If the fit is hopeless we
# fall back to upper-left, which is how these coins are usually shot. Returns a
# unit light direction.
def estimate_light(L, inside, macro_sigma=8.0, bowl_sigma=40.0, elevation=0.6):
    z = ndimage.gaussian_filter(L, macro_sigma) - ndimage.gaussian_filter(L, bowl_sigma)
    zy, zx = np.gradient(z)
    nz = 1.0 / np.sqrt(zx * zx + zy * zy + 1.0)
    nx, ny = -zx * nz, -zy * nz
    mag = np.hypot(zx, zy)

    fallback = _norm([-0.5, -0.5, elevation])
    if not inside.any():
        return fallback

    sel = inside & (mag > np.percentile(mag[inside], 55.0))
    if sel.sum() < 50:
        return fallback

    A = np.stack([np.ones(int(sel.sum())), nx[sel], ny[sel]], 1)
    try:
        coef, *_ = np.linalg.lstsq(A, L[sel], rcond=None)
        lx, ly = float(coef[1]), float(coef[2])
    except Exception:
        lx, ly = -0.5, -0.5

    n = float(np.hypot(lx, ly))
    if n < 1e-4:
        return fallback
    lx, ly = lx / n * 0.8, ly / n * 0.8
    return _norm([lx, ly, elevation])


# Surface normals from a band-passed height field (macro_sigma / bowl_sigma are
# the blurs, height_scale exaggerates the relief), with an optional dab of fine
# detail on top: fine_sigma sets the band, fine_amt its weight. Returns the
# (nx, ny, nz) normal maps.
def macro_normals(L, macro_sigma=8.0, bowl_sigma=40.0, height_scale=5.0,
                  fine_sigma=1.5, fine_amt=0.0):
    z = (ndimage.gaussian_filter(L, macro_sigma)
         - ndimage.gaussian_filter(L, bowl_sigma)) * height_scale
    if fine_amt > 0:
        z = z + fine_amt * (L - ndimage.gaussian_filter(L, fine_sigma)) * height_scale
    zy, zx = np.gradient(z)
    nz = 1.0 / np.sqrt(zx * zx + zy * zy + 1.0)
    return (-zx * nz).astype(np.float32), (-zy * nz).astype(np.float32), nz.astype(np.float32)


# Read the coin's own polished tone off its brightest pixels: anything above
# the bright_pct luminance percentile counts as a highlight. Returns an RGB tone.
def silver_tone(rgb, inside, bright_pct=80.0):
    L = _lum(rgb)
    if not inside.any():
        return np.array([0.75, 0.74, 0.70], np.float32)
    thr = np.percentile(L[inside], bright_pct)
    bright = inside & (L >= thr)
    if bright.sum() < 20:
        bright = inside
    return np.array([float(np.median(rgb[..., c][bright])) for c in range(3)], np.float32)


# Blinn-Phong specular layer. Faces pointing at the light glint; a gate driven
# by L_for_gate keeps the shine out of the coin's dark recesses (nothing shines
# below the gate_pct percentile); and if grain is set, sparse hairlines are
# carved out with rng so worn silver shines with interruptions rather than
# evenly. shininess is the specular exponent, broaden the final blur. Returns a
# float specular map.
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
        tex = np.zeros((H, W), np.float32)
        base_angle = float(rng.uniform(0.0, 180.0))
        for delta, wgt in ((0.0, 1.0), (37.0, 0.5)):
            n = ndimage.gaussian_filter(
                rng.standard_normal((H, W)).astype(np.float32), (0.5, 4.5))
            n = ndimage.rotate(n, base_angle + delta, reshape=False,
                               order=1, mode="reflect")
            n /= max(n.std(), 1e-6)
            tex += wgt * np.clip(n - 1.05, 0.0, None)
        out = out * np.clip(1.0 - grain * tex, 0.0, 1.0)

    return out.astype(np.float32)


# Hunt down the big dark blobs the wear step can leave behind and lift them
# toward the local average. Heavy wear should flatten contrast, not paint black
# smears onto the coin.
#
# worn is the image after wear and orig_mean its starting mean luminance, which
# we hold steady when keep_mean is on. lift sets how hard the blobs brighten; a
# blob has to be darker than dark_thr and at least min_area pixels to count, and
# rim_erode trims the rim first so the edge isn't mistaken for a shadow.
# field_sigma is the blur radius of the local average.
def reduce_big_shadows(worn, orig_mean, inside, lift=0.6, field_sigma=16.0,
                       dark_thr=0.36, min_area=90, rim_erode=8, keep_mean=True):
    Lw = _lum(worn)
    field = ndimage.gaussian_filter(Lw, field_sigma)
    darkness = np.clip(field - Lw, 0.0, None)
    if not inside.any():
        return worn

    hi = np.percentile(darkness[inside], 95.0)
    dn = np.clip(darkness / max(hi, 1e-6), 0.0, 1.0)
    core = ndimage.binary_erosion(inside, iterations=rim_erode)
    blobs = (dn > dark_thr) & core

    labels, count = ndimage.label(blobs)
    if count:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
        blobs = np.isin(labels, [i + 1 for i in range(count) if sizes[i] >= min_area])

    w = ndimage.gaussian_filter(blobs.astype(np.float32), 5.0)
    w = w / max(w.max(), 1e-6) * dn
    lifted = Lw + lift * w * (field - Lw)

    if keep_mean:
        m = float(lifted[inside].mean())
        if m > 1e-6:
            lifted = lifted * (orig_mean / m)

    ratio = np.clip(lifted / np.clip(Lw, 0.04, None), 0.0, 3.0)[..., None]
    return np.clip(worn * ratio, 0.0, 1.0).astype(np.float32)


# Re-light a sanded coin so it reads as handled silver rather than airbrushed:
# nudge the colour toward the coin's own metal tone, drop a specular glint on
# from the estimated light, and pull the brightness back to where it started.
#
# B is the sanded image, original the untouched one (used for the normals and
# the light estimate); if light is None it gets estimated from the photo. The
# rest tune the look: silver_boost is how hard we pull toward the metal tone,
# spec_gain / shininess / spec_color / gate_pct shape the glint, fine_amt adds
# fine detail to the normals, brightness_bias sets the target brightness,
# silver_hue is the fallback tint for near-grey coins, and spec_grain / rng
# carve the hairline scratches. Returns (relit image, light used, specular map).
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

    # learn the coin's hue from its highlights, or use a neutral silver
    tone = silver_tone(original.astype(np.float32), inside)
    hi, lo = float(tone.max()), float(tone.min())
    saturation = (hi - lo) / max(hi, 1e-6)
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
