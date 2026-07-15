from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# What a filter hands back: the altered image, a soft map of where the damage
# landed, and the coin silhouette (chipping can eat into it).
# @params image: float RGB image in [0, 1]
# @params damage_mask: float map in [0, 1], 1 = fully damaged
# @params coin_mask: the coin mask after this filter ran
# @params params: the settings that were used, kept for logging
@dataclass
class DamageResult:
    image: np.ndarray
    damage_mask: np.ndarray
    coin_mask: np.ndarray
    params: dict = field(default_factory=dict)


# Shared base so the filters can be stacked one after another.
class DamageFilter(ABC):

    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of the coin pixels
    # @params seed: seed for reproducible randomness, or None
    # @output a DamageResult
    @abstractmethod
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        ...
