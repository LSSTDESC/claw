import pyccl as ccl
from .parent import MassFunction

class CCLMassFunction(MassFunction):

    def __init__(self, hmd, hmf, cosmo):

        self.hmd = hmd
        self.hmf = hmf
        self.hmf_args = ()
        self.cosmo = cosmo

    def __call__(
        self, logM, z, *hmf_args):
        """
        parameters

        ccl_cosmo : pyccl Cosmology
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.
        reuturn
        -------

        nm : float
            Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        hmf_call = self.hmf(self.cosmo, self.hmd, *hmf_args)

        return hmf_call.get_mass_function(
            self.cosmo, 10**logM, 1.0/(1.0+z)) # pylint: disable=invalid-name


class Tinker08MassFunction(CCLMassFunction):

    def __init__(self, cosmo):

        super().__init__(
            ccl.halos.MassDef200c(),
            ccl.halos.MassFuncTinker08,
            cosmo,
            )


class Bocquet18MassFunction(CCLMassFunction):

    def __init__(self, cosmo):

        super().__init__(
            ccl.halos.MassDef200c(),
            ccl.halos.MassFuncBocquet16,
            cosmo
            )
