"""Cluster Redshift Module
abstract class to compute cluster redshift functions.
========================================
The implemented functions use PyCCL library as backend.
"""
import numpy as np
from ..utils import gaussian, gaussian_integral


class Gaussian:
    """Cluster Redshift module."""

    def __init__(
        self,
    ):
        pass

    def _get_sigmazobs(self, z, sigma_0):
        return sigma_0*(z+1.0)

    def prob_z_zobs(self, z, z_obs, sigma_0):

        sigma = self._get_sigmazobs(z, sigma_0)

        return gaussian(z_obs, z, sigma)

    def prob_z_zobs_integ(self, z, z_obs_lim, sigma_0):

        sigma = self._get_sigmazobs(z, sigma_0)
        x_min = (z - z_obs_lim[0]) / (np.sqrt(2.0) * sigma)
        x_max = (z - z_obs_lim[1]) / (np.sqrt(2.0) * sigma)

        return gaussian_integral(x_min, x_max)
