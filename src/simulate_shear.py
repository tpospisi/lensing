"""Generates ABC draws of shear maps from the prior distribution"""

import random
import sys
import galsim
from lenstools.simulations import Nicaea
import numpy as np

class Constants:
    """Class for cosmological constants"""
    def __init__(self, H0, OmegaM, Ode0, sigma8, Ob0):
        self.H0 = H0
        self.OmegaM = OmegaM
        self.Ode0 = Ode0
        self.sigma8 = sigma8
        self.Ob0 = Ob0

    def nicaea_object(self):
        return Nicaea(H0=self.H0, Om0=self.OmegaM, Ode0=self.Ode0, sigma8=self.sigma8, Ob0=self.Ob0)

    def write_to_file(self, fname):
        np.savetxt(fname, np.array([self.H0, self.OmegaM,
                                    self.Ode0,
                                    self.sigma8,
                                    self.Ob0]))

def simulate_shear(constants, redshift, noise_sd=0.0, seed=0):
    """Takes cosmological parameters, generates a shear map, and adds
    noise.

    Inputs:
        constants: Constants, object for cosmological constants
        redshift: float, the redshift value for the sample
        noise_sd: float, the standard deviation for IID Gaussian noise.
        seed: integer, seed for the RNG; if 0 it uses a randomly chosen seed.

    Returns:
        A pair of shear spectrum grids.
    """

    grid_nx = 100
    # length of grid in one dimension (degrees)
    theta = 10.0
    # grid spacing
    dtheta = theta/grid_nx

    # wavenumbers at which to evaluate power spectra
    ell = np.logspace(-2.0, 4.0, num=50)

    nicaea_obj = constants.nicaea_object()
    psObs_nicaea = nicaea_obj.convergencePowerSpectrum(ell=ell, z=redshift)
    psObs_tabulated = galsim.LookupTable(ell, psObs_nicaea, interpolant='linear')
    ps_galsim = galsim.PowerSpectrum(psObs_tabulated, delta2=False, units=galsim.radians)

    grid_deviate = galsim.BaseDeviate(seed)
    g1, g2, kappa = ps_galsim.buildGrid(grid_spacing=dtheta,
                                        ngrid=grid_nx,
                                        rng=grid_deviate,
                                        units='degrees',
                                        kmin_factor=2, kmax_factor=2,
                                        get_convergence=True)
    g1_r, g2_r, _ = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)
    rng = galsim.GaussianDeviate(seed=seed)

    g1_noise_grid = galsim.ImageD(grid_nx, grid_nx)
    g2_noise_grid = galsim.ImageD(grid_nx, grid_nx)

    g1_noise_grid.addNoise(galsim.GaussianNoise(rng=rng, sigma=noise_sd))
    g2_noise_grid.addNoise(galsim.GaussianNoise(rng=rng, sigma=noise_sd))

    g1_noisy = np.add(g1_r, g1_noise_grid.array)
    g2_noisy = np.add(g2_r, g2_noise_grid.array)

    return g1_noisy, g2_noisy

def main(outdir, true_constants, redshift, ndraws, noise_sd):
    """Simulates shear images using ABC.

    Inputs:
        outfile: string, the file to write out the images.
        true_constants: NICAEA object, the "true" cosmological constants.
        ndraws: int, the number of draws from the posterior distribution.

    Side Effects:
        Populates outdir with files g1-0.csv, g2-0.csv,
        g1-1.csv, and so on.
    """

    g1, g2 = simulate_shear(true_constants, redshift, noise_sd=noise_sd, seed=0)
    np.savetxt(outdir + "/g1-0.csv", g1)
    np.savetxt(outdir + "/g2-0.csv", g2)

    for ii in range(ndraws):
        constants = draw_constants()
        g1, g2 = simulate_shear(constants, redshift, noise_sd=noise_sd, seed=ii)
        np.savetxt(outdir + "/g1-{}.csv".format(ii + 1), g1)
        np.savetxt(outdir + "/g2-{}.csv".format(ii + 1), g2)


def draw_constants():
    OmegaM = random.uniform(0.1, 0.8)
    sigma8 = random.uniform(0.5, 1.0)

    return Constants(H0=0.70, OmegaM=OmegaM, Ode0 = 0.697,
                     sigma8=sigma8, Ob0 = 0.045)

if __name__ == '__main__':
    outdir = sys.argv[1]
    true_constants = Constants(H0=0.70, OmegaM=0.25, Ode0 = 0.697,
                               sigma8=0.8, Ob0 = 0.045)
    noise_sd = 0.00811
    redshift = 0.7
    ndraws = int(sys.argv[2])

    main(outdir, true_constants, redshift, ndraws, noise_sd)
