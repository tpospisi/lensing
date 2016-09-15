# stuff we import
import numpy as np
from lenstools.simulations import Nicaea
import galsim
import treecorr

# Global variables (for shame)
grid_nx = 100
# length of grid in one dimension (degrees)
theta = 10.
# grid spacing
dtheta = theta/grid_nx
nbins = 10
# parameters for corr2:
min_sep = dtheta # lowest bin starts at grid spacing
max_sep = grid_nx * np.sqrt(2) * dtheta # upper edge of upper bin is at maximum pair separation

## Stolen from Rachel
def run_treecorr(x, y, g1, g2):
    """Helper routine to take outputs of GalSim shear grid routine, and run treecorr on it."""
    import pyfits
    import os
    import treecorr
    # Use fits binary table for faster I/O.
    assert x.shape == y.shape
    assert x.shape == g1.shape
    assert x.shape == g2.shape
    x_col = pyfits.Column(name='x', format='1D', array=x.flatten() )
    y_col = pyfits.Column(name='y', format='1D', array=y.flatten() )
    g1_col = pyfits.Column(name='g1', format='1D', array=g1.flatten() )
    g2_col = pyfits.Column(name='g2', format='1D', array=g2.flatten() )
    cols = pyfits.ColDefs([x_col, y_col, g1_col, g2_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    hdus.writeto('temp.fits',clobber=True)
    # Define the treecorr catalog object.
    #     cat = treecorr.Catalog('temp.fits',x_units='degrees',y_units='degrees', x_col='x',y_col='y',g1_col='g1',g2_col='g2')
    cat = treecorr.Catalog('temp.fits',x_units='degrees',y_units='degrees', x_col='x',y_col='y',g1_col='g1',g2_col='g2')
    # Define the corrfunc object
    ### WAS: gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=0.1, sep_units='degrees')
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins = nbins, sep_units='degrees', bin_slop = 0.2)
    # Actually calculate the correlation function.
    gg.process(cat)
    os.remove('temp.fits')
    return gg

## ABC lensing simulations
# FOR LATER: It would be good if we had all of this stored in a config file and then had a program that read those all in.

## Big picture: ideally this is a module, which you could, in principle, swap out for a much more sophisticated black box
## WE STILL NEED TO FIGURE OUT HOW TO ADD SHAPE NOISE APPROPRIATELY TO THESE ESTIMATES
## SENSIBLE THING SEEMS TO BE TO PASS IN S/N AS A PARAMETERg
## Function simulateShear

# Takes cosmological parameters, generates a shear map, adds noise, computes binned CF summary statistics and returns them...
# Inputs (for now): OmegaM_arg, sigma8_arg, noise_arg, seed
# if seed_arg = 0 (the default), then it should use "a random seed from the system" per the documentation
# noise_arg is, for now, just the standard deviation for the IID Gaussian
def simulate_shear(OmegaM_arg, sigma8_arg, noise_arg, seed_arg = 0):
    
    ## Parameters that are fixed, for now (i.e., the prior is a Dirac delta function!)
    H0Fix = 70.0
    Ob0Fix = 0.045
    z_use = 0.7
    
    # wavenumbers to evaluate power spectra at
    ell_use = np.logspace(-2.0, 4.0, num = 50)

    # Force that Om0 + Ode0 = 1
    nicaeaObs = Nicaea(H0 = H0Fix, Om0 = OmegaM_arg, Ode0 = 1-OmegaM_arg, sigma8 = sigma8_arg, Ob0=Ob0Fix)
    psObs_nicaea = nicaeaObs.convergencePowerSpectrum(ell = ell_use, z = z_use)
    psObs_tabulated = galsim.LookupTable(ell_use, psObs_nicaea, interpolant = 'linear')
    ps_galsim = galsim.PowerSpectrum(psObs_tabulated, delta2 = False, units = galsim.radians)
    
    grid_deviate = galsim.BaseDeviate(seed_arg)
    #    g1, g2, kappa = ps_galsim.buildGrid(grid_spacing=dtheta, ngrid=grid_nx, rng=grid_deviate, units='degrees', kmin_factor=2, kmax_factor=2)
    g1, g2, kappa = ps_galsim.buildGrid(grid_spacing=dtheta, ngrid=grid_nx, rng=grid_deviate, units='degrees', kmin_factor=2, kmax_factor=2, get_convergence = True)
    g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)
    rng = galsim.GaussianDeviate(lseed = seed_arg)

    g1_noise_grid = galsim.ImageD(grid_nx, grid_nx)
    g2_noise_grid = galsim.ImageD(grid_nx, grid_nx)
    g1_noise_grid.addNoise(galsim.GaussianNoise(rng = rng, sigma = noise_arg))
    g2_noise_grid.addNoise(galsim.GaussianNoise(rng = rng, sigma = noise_arg))
    
    g1_noisy = np.add(g1_r, g1_noise_grid.array)
    g2_noisy = np.add(g2_r, g2_noise_grid.array)

    grid_range = dtheta * np.arange(grid_nx)
    x, y = np.meshgrid(grid_range, grid_range)
    gg = run_treecorr(x, y, g1_noisy, g2_noisy)
    
    to_return = {'log_r' : gg.logr, 'xipm' : np.hstack((gg.xip,gg.xim))}
    #    to_return = np.hstack((gg.xip,gg.xim))
    return(to_return)
