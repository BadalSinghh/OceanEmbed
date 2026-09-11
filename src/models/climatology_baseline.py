"""
climatology_baseline.py

Climatology / Depth Baseline Model for subsurface ocean temperature.
Predicts historical temporal/spatial mean profile across requested 15 depths.
"""

import numpy as np

class ClimatologyBaseline:
    """
    Computes spatial-temporal mean temperature profiles across grid points and depths.
    """
    def __init__(self, target_depths=None):
        self.target_depths = target_depths or [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.climatology_mean = None

    def fit(self, target_tensor: np.ndarray):
        """
        target_tensor shape: (N_time, N_depths, N_lat, N_lon)
        """
        # Mean across time dimension
        self.climatology_mean = np.nanmean(target_tensor, axis=0, keepdims=True)
        return self

    def predict(self, num_samples: int):
        """
        Broadcasts climatology mean for num_samples timesteps.
        Returns shape: (num_samples, N_depths, N_lat, N_lon)
        """
        if self.climatology_mean is None:
            raise ValueError("Climatology model is not fitted yet.")
        return np.repeat(self.climatology_mean, num_samples, axis=0)
