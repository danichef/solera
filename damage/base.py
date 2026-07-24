from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# What a filter hands back: the altered image, a soft map of where the damage
# landed, and the coin silhouette (chipping can eat into it).
@dataclass
class DamageResult:
    image: np.ndarray                              # float RGB in [0, 1]
    damage_mask: np.ndarray                        # float map in [0, 1], 1 = fully damaged
    coin_mask: np.ndarray                          # the coin mask after this filter ran
    params: dict = field(default_factory=dict)     # settings used, kept for logging


# Shared base so the filters can be stacked one after another.
class DamageFilter(ABC):

    # Run the filter on one face. image is a float RGB image in [0, 1],
    # coin_mask marks the coin pixels, and seed makes the randomness
    # reproducible (pass None to get fresh randomness each time).
    @abstractmethod
    def apply(self, image, coin_mask, seed=None) -> DamageResult:
        ...
