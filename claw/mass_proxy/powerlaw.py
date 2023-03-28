"""Cluster Mass Richness proxy
"""
import numpy as np
from ..utils import gaussian, gaussian_integral

ln10 = np.log(10)

class PowerLawGaussian:
    """Cluster Mass Richness proxy."""

    def __init__(
        self,
        logM_pivot,
        z_pivot,
    ):
        self.logM_pivot = logM_pivot
        self.log_z_p1_pivot = np.log10(1.0 + z_pivot)

    def _get_meanlogMobs_sigmalogMobs(
        self, logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
    ):
        log_z_p1 = np.log10(1.0+z)
        lnM_obs_mean = (
            mu_p0
            + mu_p1 * (logM - self.logM_pivot) * ln10
            + mu_p2 * (log_z_p1 - self.log_z_p1_pivot) * ln10
        )
        sigma = (
            sigma_p0
            + sigma_p1 * (logM - self.logM_pivot) * ln10
            + sigma_p2 * (log_z_p1 - self.log_z_p1_pivot) * ln10
        )
        # sigma = abs(sigma)
        return lnM_obs_mean/ln10, sigma/ln10

    def prob_logM_logMobs(
        self, logM, logM_obs, z,
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
    ):
        logM_obs_mean, sigma = self._get_meanlogMobs_sigmalogMobs(
            logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2)

        return gaussian(logM_obs, logM_obs_mean, sigma)

    def prob_logM_logMobs_integ(
        self, logM, logM_obs_lim, z,
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2,
    ):
        logM_obs_mean, sigma = self._get_meanlogMobs_sigmalogMobs(
            logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2)
        x_min = (logM_obs_mean - logM_obs_lim[0]) / (np.sqrt(2.0) * sigma)
        x_max = (logM_obs_mean - logM_obs_lim[1]) / (np.sqrt(2.0) * sigma)

        return gaussian_integral(x_min, x_max)
