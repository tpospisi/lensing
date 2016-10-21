"""Generates ABC draws of shear maps from the prior distribution"""

import random
import sys
import galsim
from lenstools.simulations import Nicaea
import numpy as np

class Constants:
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


def draw_cosmological_constants(H0=None, OmegaM=None, Ode0=None, sigma8=None, Ob0=None):
  """Draws a set of cosmological constants from the prior distributions

  Inputs:
    H0: float, a cosmological constant
    OmegaM: float, a cosmological constant
    Ode0: float, a cosmological constant
    sigma8: float, a cosmological constant
    Ob0: float, a cosmological constant

  Returns:
    A NICAEA object with the constants set.
  """

  if H0 is None:
    H0 = 70.0
  if OmegaM is None:
    OmegaM = 0.2
  if Ode0 is None:
    Ode0 = 1 - OmegaM # Force that Om0 + Ode0 = 1
  if sigma8 is None:
    sigma8 = random.uniform(0, OmegaM)
  if Ob0 is None:
    Ob0 = 0.045

  return Constants(H0=H0, OmegaM=OmegaM, Ode0=Ode0, sigma8=sigma8, Ob0=Ob0)


def simulate_shear(constants=None, noise_sd=0.0, seed=0):
  """Takes cosmological parameters, generates a shear map, and adds
  noise.

  Inputs:
    OmegaM: float, cosmological constant
    sigma8: float, cosmological constant
    noise_sd: float, the standard deviation for IID Gaussian noise.
    seed: integer, seed for the RNG; if 0 it uses a randomly chosen seed.

  Returns:
    A pair of shear spectrum grids.
  """

  grid_nx = 100
  # length of grid in one dimension (degrees)
  theta = 10.
  # grid spacing
  dtheta = theta/grid_nx

  redshift = 0.7

  # wavenumbers at which to evaluate power spectra
  ell = np.logspace(-2.0, 4.0, num=50)

  if constants is None:
      constants = draw_cosmological_constants()
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

def main(outdir, true_constants, noise_sd, ndraws, epsilon):
  """Simulates shear images using ABC.

  Inputs:
    outfile: string, the file to write out the images.
    true_constants: NICAEA object, the "true" cosmological constants.
    noise_sd: float, the standard deviation of added IID Gaussian noise.
    ndraws: int, the number of draws from the posterior distribution.
    epsilon: float, the threshold for ABC selection.

  Side Effects:
    Populates outdir with files g1-1.csv, g2-1.csv,
    g1-2.csv, and so on.
  """
  true_g1, true_g2 = simulate_shear(true_constants, noise_sd=0.0, seed=1)

  true_constants.write_to_file(outdir + "/params-0.csv")
  np.savetxt(outdir + "/g1-0.csv", true_g1)
  np.savetxt(outdir + "/g2-0.csv", true_g2)

  for ii in xrange(ndraws):
    print "Iteration {}\n".format(ii)
    while True:
      draw = draw_cosmological_constants()
      g1, g2 = simulate_shear(draw, seed=ii)
      if np.linalg.norm(true_g1 - g1) < epsilon and np.linalg.norm(true_g2 - g2) < epsilon:
        draw.write_to_file(outdir + "/constants-{}.csv".format(ii + 1))
        np.savetxt(outdir + "/g1-{}.csv".format(ii + 1), g1)
        np.savetxt(outdir + "/g2-{}.csv".format(ii + 1), g2)
        break


if __name__ == '__main__':
  outdir = sys.argv[1]
  true_constants = draw_cosmological_constants(sigma8=0.2)
  noise_sd = 0.0
  ndraws = int(sys.argv[2])
  epsilon = float(sys.argv[3])

  main(outdir, true_constants, noise_sd, ndraws, epsilon)
