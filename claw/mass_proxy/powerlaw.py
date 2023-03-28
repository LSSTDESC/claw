"""Cluster Mass Richness proxy
"""
import pyccl as ccl
import numpy as np
from scipy import special
from .cluster_mass import ClusterMass

ln10 = np.log(10)

class PowerLawGaussian:
    """Cluster Mass Richness proxy."""

    def __init__(
        self,
        pivot_mass,
        pivot_redshift,
    ):
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.log_pivot_mass = np.log10(10**pivot_mass)
        self.log_pivot_redshift_p1 = np.log10(1.0 + self.pivot_redshift)

    def _get_meanlogMobs_sigmalogMobs(
        self, logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
    ):
        log_z_p1 = np.log10(1.0+z)
        lnM_obs_mean = (
            mu_p0
            + mu_p1 * (logM - self.log_pivot_mass) * ln10
            + mu_p2 * (log_z_p1 - self.log_pivot_redshift_p1) * ln10
        )
        sigma = (
            sigma_p0
            + sigma_p1 * (logM - self.log_pivot_mass) * ln10
            + sigma_p2 * (log_z_p1 - self.log_pivot_redshift_p1) * ln10
        )
        # sigma = abs(sigma)
        return lnM_obs_mean/ln10, sigma/ln10

    def prob_logM_logMobs(
        self, logM, logM_obs, z,
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
    ):
        logM_obs_mean, sigma = self._get_meanlogMobs_sigmalogMobs(
            logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2)
        chisq = (logM_obs-logM_obs_mean)**2 / (2*sigma**2)
        return np.exp(-chisq) / (np.sqrt(2.0 * np.pi) * sigma)

    def prob_logM_logMobs_integ(
        self, logM, logM_obs_lim,
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2,
    ):
        logM_obs_mean, sigma = self._get_meanlogMobs_sigmalogMobs(
            logM, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2)
        x_min = (logM_obs_mean - logM_obs_lim[0]) / (np.sqrt(2.0) * sigma)
        x_max = (logM_obs_mean - logM_obs_lim[1]) / (np.sqrt(2.0) * sigma)
        if x_max > 4.0:
            return (special.erfc(x_min) - special.erfc(x_max)) / 2.0
        else:
            return (special.erf(x_min) - special.erf(x_max)) / 2.0
