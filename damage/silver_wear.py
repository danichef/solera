from __future__ import annotations

import numpy as np

from .base import DamageFilter, DamageResult
from .sanding_classic import ClassicSandingFilter
from . import silver_relight as relight


# The full wear stage: runs the sanding simulation and then re-lights the
# result so worn areas gleam like handled silver instead of looking airbrushed.
class SilverWearFilter(DamageFilter):

    # @params grade: preset name passed on to the sanding filter
    # @params spec_gain / shininess / gate_pct: specular glint settings
    # @params silver_boost: pull toward the coin's metal tone
    # @params fine_amt: fine detail in the shading normals
    # @params shadow_lift: strength of the dark-blob cleanup
    # @params brightness_bias: target brightness relative to the original
    # @params light: fixed light direction, estimated from the photo when None
    # @params spec_grain: hairline scratches carved out of the shine
    def __init__(self, grade="vg", spec_gain=2.1, shininess=10.0,
                 silver_boost=0.55, fine_amt=0.25, gate_pct=45.0,
                 shadow_lift=0.6, brightness_bias=1.0, light=None,
                 spec_grain=0.0):
        self.grade = grade
        self.spec_gain = spec_gain
        self.shininess = shininess
        self.silver_boost = silver_boost
        self.fine_amt = fine_amt
        self.gate_pct = gate_pct
        self.shadow_lift = shadow_lift
        self.brightness_bias = brightness_bias
        self.light = light
        self.spec_grain = spec_grain

    _GRADES = {
        "vf": dict(grade="vf", spec_gain=1.6, shininess=12.0, silver_boost=0.45, shadow_lift=0.40, gate_pct=46.0, spec_grain=0.25),
        "f":  dict(grade="f",  spec_gain=1.9, shininess=11.0, silver_boost=0.50, shadow_lift=0.45, gate_pct=48.0, spec_grain=0.32),
        "vg": dict(grade="vg", spec_gain=2.2, shininess=10.0, silver_boost=0.55, shadow_lift=0.50, gate_pct=50.0, spec_grain=0.40),
        "g":  dict(grade="g",  spec_gain=2.4, shininess=9.0,  silver_boost=0.58, shadow_lift=0.54, gate_pct=52.0, spec_grain=0.48),
        "ag": dict(grade="ag", spec_gain=2.7, shininess=8.0,  silver_boost=0.62, shadow_lift=0.56, gate_pct=54.0, spec_grain=0.55),
    }

    # Builds the filter from a named grade preset, with optional overrides.
    # @params grade: one of vf / f / vg / g / ag
    # @output configured SilverWearFilter
    @classmethod
    def for_grade(cls, grade: str, **overrides):
        config = dict(cls._GRADES[grade])
        config.update(overrides)
        return cls(**config)

    # Wears the coin down and re-lights it.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of coin pixels
    # @params seed: seed shared by the sanding and relighting randomness
    # @output DamageResult carrying the sanding wear mask
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        img = image.astype(np.float32)
        inside = coin_mask > 0.5
        cm = coin_mask[..., None]
        if inside.any():
            original_mean = float(relight._lum(img)[inside].mean())
        else:
            original_mean = 0.5

        sanded = ClassicSandingFilter.for_grade(self.grade).apply(image, coin_mask,
                                                                  seed=seed)

        rng = np.random.default_rng(None if seed is None else seed + 101)
        out, light, spec = relight.silverize_and_shine(
            sanded.image, img, inside, light=self.light,
            silver_boost=self.silver_boost, spec_gain=self.spec_gain,
            shininess=self.shininess, gate_pct=self.gate_pct,
            fine_amt=self.fine_amt, brightness_bias=self.brightness_bias,
            spec_grain=self.spec_grain, rng=rng)
        out = relight.reduce_big_shadows(out, original_mean, inside,
                                         lift=self.shadow_lift)

        out = np.clip(out * cm + image * (1.0 - cm), 0.0, 1.0).astype(np.float32)
        params = dict(filter="silver_wear", seed=seed, grade=self.grade,
                      spec_gain=self.spec_gain, silver_boost=self.silver_boost,
                      shadow_lift=self.shadow_lift,
                      light=tuple(round(float(x), 3) for x in light))
        return DamageResult(out, sanded.damage_mask, coin_mask, params)
