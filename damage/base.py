from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# Carries the output of one damage filter: the modified image, a soft mask of
# where damage was applied, and the (possibly shrunken) coin silhouette.
# @params image: float RGB image in [0, 1]
# @params damage_mask: float map in [0, 1], 1 = fully damaged
# @params coin_mask: float mask of the coin after this filter
# @params params: settings the filter used, for logging
@dataclass
class DamageResult:
    image: np.ndarray
    damage_mask: np.ndarray
    coin_mask: np.ndarray
    params: dict = field(default_factory=dict)


# Base class for damage filters so they can be chained: each takes an image
# plus coin mask and returns a DamageResult.
class DamageFilter(ABC):

    # Applies the filter to one coin face.
    # @params image: float RGB image in [0, 1]
    # @params coin_mask: float mask of coin pixels
    # @params seed: optional seed for reproducible randomness
    # @output DamageResult
    @abstractmethod
    def apply(self, image: np.ndarray, coin_mask: np.ndarray,
              seed: int | None = None) -> DamageResult: ...
