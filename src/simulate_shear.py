#!/usr/bin/python3

"""Generates ABC draws of shear maps from the prior distribution"""

import csv
import os
import random
import sys

import galsim
from lenstools.simulations import Nicaea
import numpy as np
import pyfits
import treecorr

class Constants:
    """Class for cosmological constants"""
    def __init__(self, H0, omega_m, Ode0, sigma_8, Ob0):
        self.H0 = H0
        self.omega_m = omega_m
        self.Ode0 = Ode0
        self.sigma_8 = sigma_8
        self.Ob0 = Ob0

    def nicaea_object(self):
        return Nicaea(H0=self.H0, Om0=self.omega_m, Ode0=self.Ode0,
                      sigma8=self.sigma_8, Ob0=self.Ob0)

def run_treecorr(x, y, g1, g2, min_sep, max_sep, nbins):
    """Run treecorr on GalSim shear grid routine"""

    assert x.shape == y.shape
    assert x.shape == g1.shape
    assert x.shape == g2.shape

    x_col = pyfits.Column(name='x', format='1D', array=x.flatten())
    y_col = pyfits.Column(name='y', format='1D', array=y.flatten())
    g1_col = pyfits.Column(name='g1', format='1D', array=g1.flatten())
    g2_col = pyfits.Column(name='g2', format='1D', array=g2.flatten())
    cols = pyfits.ColDefs([x_col, y_col, g1_col, g2_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu, table])
    hdus.writeto('temp.fits', clobber=True)

    # Define the treecorr catalog object.
    cat = treecorr.Catalog('temp.fits', x_units='degrees', y_units='degrees',
                           x_col='x', y_col='y', g1_col='g1', g2_col='g2')
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                sep_units='degrees', bin_slop=0.2)
    gg.process(cat)
    os.remove('temp.fits')

    return {'log_r' : gg.logr,
            'xipm' : np.hstack((gg.xip, gg.xim))}


def simulate_shear(constants, redshift, nbins, noise_sd=0.0, seed=0):
    """Takes cosmological parameters, generates a shear map, and adds
    noise.

    Inputs:
        constants: Constants, object for cosmological constants
        redshift: float, the redshift value for the sample
        nbins: int, the number of bins for the correlation function
        noise_sd: float, the standard deviation for IID Gaussian noise.
        seed: integer, seed for the RNG; if 0 it uses a randomly chosen seed.

    Returns:
        A pair of shear spectrum grids.
    """

    grid_nx = 100 # length of grid in one dimension (degrees)
    theta = 10.0 # grid spacing
    dtheta = theta / grid_nx

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

    min_sep = dtheta
    max_sep = grid_nx * np.sqrt(2) * dtheta
    grid_range = dtheta * np.arange(grid_nx)
    x, y = np.meshgrid(grid_range, grid_range)
    stats = run_treecorr(x, y, g1, g2, min_sep, max_sep, nbins=nbins)

    return g1_noisy, g2_noisy, stats

def main(outfile, true_constants, redshift, ndraws, noise_sd, nbins):
    """Simulates shear images using ABC.

    Inputs:
        outfile: string, the file to write out the images.
        true_constants: NICAEA object, the "true" cosmological constants.
        ndraws: int, the number of draws from the posterior distribution.

    Side Effects:
        Populates outdir with files g1-0.csv, g2-0.csv,
        g1-1.csv, and so on.
    """

    with open(outfile, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["sigma_8", "omega_m"] + ["X" + str(ii + 1) for ii in range(nbins * 2)])

        g1, g2, stats = simulate_shear(true_constants, redshift, nbins, noise_sd=noise_sd, seed=0)
        write_to_csv(writer, true_constants, g1, g2, stats)

        for sim in range(1, ndraws + 1):
            constants = draw_constants()
            g1, g2, stats = simulate_shear(constants, redshift, nbins, noise_sd=noise_sd, seed=sim)

            write_to_csv(writer, constants, g1, g2, stats)


def write_to_csv(writer, constants, g1, g2, stats):
    writer.writerow([constants.sigma_8, constants.omega_m] + [xx for xx in stats["xipm"]])

def draw_constants():
    """ Draws cosmological constants from prior distribution"""

    omega_m = random.uniform(0.1, 0.8)
    sigma_8 = random.uniform(0.5, 1.0)

    return Constants(H0=0.70, omega_m=omega_m, Ode0=0.697,
                     sigma_8=sigma_8, Ob0=0.045)

if __name__ == '__main__':
    outdir = sys.argv[1]
    true_constants = Constants(H0=0.70, omega_m=0.25, Ode0 = 0.697,
                               sigma_8=0.8, Ob0 = 0.045)
    noise_sd = 0.00811
    redshift = 0.7
    ndraws = int(sys.argv[2])
    nbins = int(sys.argv[3])

    main(outdir, true_constants, redshift, ndraws, noise_sd, nbins)
