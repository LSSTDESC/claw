r"""Cluster Object
"""
import numpy as np
import scipy.integrate


class ClusterAbundance():
    r"""Cluster Abundance Object.
        Atributes
        ---------
        cluster_mass: ClusterMass object
            Dictates whether to use a mass proxy or not,\
            which mass function and other cluster functions\
            that mostly depend on the cluster mass.
        cluster_redshift: Cluster Redshift object
            Dictates whether to use a redshift proxy or not,\
            how to compute the comoving volume and other cluster functions\
            that mostly depend on the cluster redshift.
        sky_area: float
            Area of the sky from the survey.
    """
    def __init__(
        self,
        halo_mass_function,
        ccl_cosmology,
        prob_logMobs_logM=None,
        prob_zobs_ztrue=None,
        prob_logMobs_logM_integ=None,
        prob_zobs_ztrue_integ=None,
    ):

        self.hmf = halo_mass_function
        self.cosmo = ccl_cosmology

        self.has_mass_proxy = any(
            prob is not None for prob in (
                prob_logMobs_logM,
                prob_logMobs_logM_integ)
            )
        self.has_redshift_proxy = any(
            prob is not None for prob in (
                prob_zobs_ztrue,
                prob_zobs_ztrue_integ)
            )

        if self.has_mass_proxy:
            self.prob_logMobs_logM = prob_logMobs_logM
            self.prob_logMobs_logM_integ = (
                prob_logMobs_logM_integ
                if prob_logMobs_logM_integ is not None
                else self._prob_logMobs_logM_integ)
        else:
            self.prob_logMobs_logM = lambda *args: raise NotImplementedError(
                "There is no P(logM_obs|logM) defined!")
            self.prob_logMobs_logM_integ = lambda *args: 1.0

        if self.has_redshift_proxy:
            self.prob_zobs_ztrue = prob_zobs_ztrue
            self.prob_zobs_z_integ = (
                prob_zobs_z_integ if prob_zobs_z_integ is not None
                else self._prob_zobs_z_integ)
        else:
            self.prob_zobs_ztrue = lambda *args: raise NotImplementedError(
                "There is no P(z_obs|z_true) defined!")
            self.prob_zobs_ztrue_integ = lambda *args: 1.0

        # used for 0->infty integrals
        self.logM_true_lim=(13.0, 16.0)
        self.z_true_lim=(0.0, 2.0),


    def compute_differential_comoving_volume(self, z):
        """
        Computes differential comoving volume

        Parameters
        ----------
        z : float
            Cluster Redshift.

        reuturn
        -------
        dv : float
            Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        a = 1.0 / (1.0 + z)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        da = ccl.background.angular_diameter_distance(ccl_cosmo, a)
        E = ccl.background.h_over_h0(ccl_cosmo, a)  # pylint: disable=invalid-name
        dV = (  # pylint: disable=invalid-name
            ((1.0 + z) ** 2)
            * (da**2)
            * ccl.physical_constants.CLIGHT_HMPC
            / ccl_cosmo["h"]
            / E
        )
        return dV


    def _prob_logMobs_logM_integ(
        self, logM, logMobs_lim, logMobs_args=(),
        epsabs=1.0e-4, epsrel=1.0e-4,
    ):
        r"""Computes the integral of P(logM_obs|logM) over observed mass limits.

        .. math::
            \int_{logM_obs_min}^{logM_obs_max} dlogM_obs \;  P(logM_obs|logM, args).

        Parameters
        ----------
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logMobs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        zobs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        return scipy.integrate.quad(
            lambda logMobs: self.prob_logMobs_logM(
                logM, logMobs, *logMobs_args),
            logMobs_lim[0],
            logMobs_lim[1],
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]

    def _prob_zobs_z_integ(
        self, z, zobs_lim, zobs_args=(),
        epsabs=1.0e-4, epsrel=1.0e-4,
    ):
        r"""Computes the integral of P(z_obs|z) over observed redshift limits.

        .. math::
            \int_{z_obs_min}^{z_obs_max} dz_obs \;  P(z_obs|z, args).

        Parameters
        ----------
        ccl_cosmo: Cosmology
            Pyccl cosmology
        zobs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        zobs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        return scipy.integrate.quad(
            lambda zobs: self.prob_zobs_z(
                z, zobs, *zobs_args),
            zobs_lim[0],
            zobs_lim[1],
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]

    def d2ndlogmdz_dvdz_fmobs_fzobs(
        self, logM, z, logMobs_lim, zobs_lim,
        hmf_args=(), logMobs_args=(), zobs_args=(),
    ):
        r"""Computes the integral of $d2n(logM, logM_obs, z, z_obs)$ over

        .. math::
            \frac{d^2n}{dlogMdz}\frac{dv}{dz}
            \int_{logM_obs_min}^{logM_obs_max} dlogM_obs \;  P(logM_obs|logM, args)
            \int_{z_obs_min}^{z_obs_max} dz \;  P(z_obs|z, args).

        Parameters
        ----------
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logMobs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        zobs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        return (
            self.compute_differential_comoving_volume(z)
            *self.hmf(logM, z, *hmf_args)
            *self.prob_logMobs_logM_integ(
                logM, logMobs_lim, logMobs_args)
            *self.prob_zobs_ztrue_integ(
                z, zobs_lim, zobs_args)
        )

    def compute_abundance(
        self, logMobs_lim, z,
        logMobs_args=(), zobs_args=(),
        epsabs=1.0e-4, epsrel=1.0e-4,
        ):
        r"""Computes the integral of $d2n(logM, logM_obs, z, z_obs)$ over
        the true values of mass and redshift, that is
        .. math::
            \int_{-\infty}^{\infty} dlogM
            \int_{0}^{\infty} dz \;
            \frac{d^2n}{dlogMdz}\frac{dv}{dz}
            \int_{logM_obs_min}^{logM_obs_max} dlogM_obs \;  P(logM_obs|logM, args)
            \int_{z_obs_min}^{z_obs_max} dz_obs \;  P(z_obs|z, args).

        Parameters
        ----------
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logMobs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        zobs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """
        if self.has_mass_proxy:
            logM_integ_lim = self.logM_true_lim
        else:
            logM_integ_lim = logMobs_lim
        if self.has_redshift_proxy:
            z_integ_lim = self.z_true_lim
        else:
            z_integ_lim = zobs_lim
        return scipy.integrate.dblquad(
            lambda logM, z: self.d2ndlogmdz_dvdz_fmobs_fzobs(
                logM, z, logMobs_lim, zobs_lim, hmf_args=hmf_args,
                logMobs_args=logMobs_args, zobs_args=zobs_args),
            z_integ_lim[0],
            z_integ_lim[1],
            # pylint: disable-next=cell-var-from-loop
            lambda x: logM_integ_lim[0],
            # pylint: disable-next=cell-var-from-loop
            lambda x: logM_integ_lim[1],
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]


    # second approach

    def d2ndlogmdz_dvdz_pmobs_pzobs(
        self, logM, z, logMobs, zobs,
        hmf_args=(), logMobs_args=(), zobs_args=(),
    ):
        r"""Define integrand with proxy for redshift and mass.

        The integrand is given by
        .. math::
            d2n(logM, logM_obs, z, z_obs) = \frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM dz.

        Parameters
        ----------
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.
        logMobs: float
            Cluster mass proxy.
        zobs: float
            Cluster Redshift proxy.
        hmf_args: tuple
            Other arguments for hmf
        logMobs_args: tuple
            Other arguments for logMobs
        zobs_args: tuple
            Other arguments for zobs

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        return (
            self.compute_differential_comoving_volume(z)
            *self.hmf(logM, z, *hmf_args)
            *self.prob_logMobs_logM(logM, logMobs, *logMobs_args)
            *self.prob_zobs_ztrue(z, zobs, *zobs_args)

    def d2ndlogmdz_dvdz_mobs_zobs(
        self, logMobs, zobs,
        hmf_args=(), logMobs_args=(), zobs_args=(),
        epsabs=1.0e-4, epsrel=1.0e-4,
    ):
        r"""Computes the integral of $d2n(logM, logM_obs, z, z_obs)$ over
        the true values of mass and redshift, that is
        .. math::
            d2n(logM_obs, z_obs) = \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM dz.

        Parameters
        ----------
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logMobs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        zobs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        return scipy.integrate.dblquad(
            lambda logM, z: self.d2ndlogmdz_dvdz_pmobs_pzobs(
                logM, z, logMobs, zobs, hmf_args=hmf_args,
                logMobs_args=logMobs_args, zobs_args=zobs_args),
            self.z_true_lim[0],
            self.z_true_lim[1],
            # pylint: disable-next=cell-var-from-loop
            lambda x: self.logM_true_lim[0],
            # pylint: disable-next=cell-var-from-loop
            lambda x: self.logM_true_lim[1],
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]
