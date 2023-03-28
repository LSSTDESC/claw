import numpy as np
from scipy import special

def gaussian(x, mu, sigma):
    return np.exp(-0.5*(x-mu)**2/sigma**2) / (np.sqrt(2.0 * np.pi) * sigma)


def gaussian_integral(x_min, x_max):
    if x_max > 4.0:
        return (special.erfc(x_min) - special.erfc(x_max)) / 2.0
    else:
        return (special.erf(x_min) - special.erf(x_max)) / 2.0
