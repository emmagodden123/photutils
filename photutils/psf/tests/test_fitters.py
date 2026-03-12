# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for PSF fitters.
"""

import numpy as np
from numpy.testing import assert_allclose

from photutils.psf.epsf import EPSFFitter
from photutils.psf.epsf_stars import EPSFStar, EPSFStars
from photutils.psf.fitters import PriorLogTRFLSQFitter
from photutils.psf.image_models import ImagePSF


def test_prior_log_trf_lsq_fitter_masks_nonpositive_weights():
    yy, xx = np.indices((9, 9), dtype=float)
    x0 = y0 = 4.0
    sigma = 1.2
    psf_data = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2.0 * sigma**2))
    psf_data /= np.sum(psf_data)

    epsf = ImagePSF(data=psf_data)
    true_flux = 100.0
    clean_data = true_flux * psf_data

    masked_outlier_data = clean_data.copy()
    masked_outlier_data[0, 0] = 1.0e6
    weights = np.ones_like(masked_outlier_data)
    weights[0, 0] = 0.0

    fitter = EPSFFitter(fitter=PriorLogTRFLSQFitter(max_nfev=1000),
                        fit_boxsize=None)

    clean_star = EPSFStar(clean_data, cutout_center=(x0, y0))
    clean_flux = fitter(epsf, EPSFStars([clean_star])).flux

    masked_star = EPSFStar(masked_outlier_data, weights=weights,
                           cutout_center=(x0, y0))
    masked_flux = fitter(epsf, EPSFStars([masked_star])).flux

    assert_allclose(masked_flux, clean_flux, rtol=5.0e-5, atol=0.0)
