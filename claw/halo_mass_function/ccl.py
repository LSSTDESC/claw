import pyccl as ccl
from .parent import MassFunction

class CCLMassFunction(MassFunction):

    def __init__(self, hmd, hmf):

        self.hmd = hmd
        self.hmf = hmf
        self.hmf_args = ()
        self.hmf_call = self.hmf(ccl_cosmo, self.hmd)

    def __call__(
        self, ccl_cosmo, logM, z, *hmf_args):
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
        if self.hmf_args!=hmf_args:
            self.hmf_call = self.hmf(ccl_cosmo, self.hmd, *hmf_args)

        return self.hmf_call.get_mass_function(
            ccl_cosmo, 10**logM, 1.0/(1.0+z)) # pylint: disable=invalid-name


class TinkerMassFunction(CCLMassFunction):

    def __init__(self):

        super().__init__(
            ccl.halos.MassDef200c(),
            ccl.halos.MassFuncTinker08,
            )


class TinkerMassFunction(CCLMassFunction):

    def __init__(self):

        super().__init__(
            ccl.halos.MassDef200c(),
            ccl.halos.MassFuncBocquet16,
            )
