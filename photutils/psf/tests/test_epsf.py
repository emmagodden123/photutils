# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf module.
"""

import itertools

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import (InverseVariance, NDData, StdDevUncertainty,
                            VarianceUncertainty)
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets import make_model_image
from photutils.psf import CircularGaussianPRF, make_psf_model_image
from photutils.psf.epsf import EPSFBuilder, EPSFFitter, PPEMap
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, extract_stars
from photutils.psf.image_models import ImagePSF
from photutils.utils._optional_deps import HAS_MATPLOTLIB


@pytest.fixture
def epsf_test_data():
    """
    Create a simulated image for testing.
    """
    fwhm = 2.7
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    model_shape = (9, 9)
    n_sources = 100
    shape = (750, 750)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=25,
                                             border_size=25, seed=0)

    nddata = NDData(data)
    init_stars = Table()
    init_stars['x'] = true_params['x_0'].astype(int)
    init_stars['y'] = true_params['y_0'].astype(int)

    return {
        'fwhm': fwhm,
        'data': data,
        'nddata': nddata,
        'init_stars': init_stars,
    }


class TestEPSFBuild:

    def test_extract_stars(self, epsf_test_data):
        size = 25
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'],
                              size=size)

        assert len(stars) == len(epsf_test_data['init_stars'])
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)

    def test_extract_stars_uncertainties(self, epsf_test_data):
        rng = np.random.default_rng(0)
        shape = epsf_test_data['nddata'].data.shape
        error = np.abs(rng.normal(loc=0, scale=1, size=shape))
        uncertainty1 = StdDevUncertainty(error)
        uncertainty2 = uncertainty1.represent_as(VarianceUncertainty)
        uncertainty3 = uncertainty1.represent_as(InverseVariance)
        ndd1 = NDData(epsf_test_data['nddata'].data, uncertainty=uncertainty1)
        ndd2 = NDData(epsf_test_data['nddata'].data, uncertainty=uncertainty2)
        ndd3 = NDData(epsf_test_data['nddata'].data, uncertainty=uncertainty3)

        size = 25
        match = 'were not extracted because their cutout region extended'
        ndd_inputs = (ndd1, ndd2, ndd3)

        outputs = [extract_stars(ndd_input, epsf_test_data['init_stars'],
                                 size=size) for ndd_input in ndd_inputs]

        for stars in outputs:
            assert len(stars) == len(epsf_test_data['init_stars'])
            assert isinstance(stars, EPSFStars)
            assert isinstance(stars[0], EPSFStars)
            assert stars[0].data.shape == (size, size)
            assert stars[0].weights.shape == (size, size)

        assert_allclose(outputs[0].weights, outputs[1].weights)
        assert_allclose(outputs[0].weights, outputs[2].weights)

        uncertainty = StdDevUncertainty(np.zeros(shape))
        ndd = NDData(epsf_test_data['nddata'].data, uncertainty=uncertainty)

        match = 'One or more weight values is not finite'
        with pytest.warns(AstropyUserWarning, match=match):
            stars = extract_stars(ndd, epsf_test_data['init_stars'][0:3],
                                  size=size)

    @pytest.mark.parametrize('shape', [(25, 25), (19, 25), (25, 19)])
    def test_epsf_build(self, epsf_test_data, shape):
        """
        This is an end-to-end test of EPSFBuilder on a simulated image.
        """
        oversampling = 2
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'][:10],
                              size=shape)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=5,
                                   progress_bar=False, norm_radius=10,
                                   recentering_maxiters=5)
        epsf, fitted_stars = epsf_builder(stars)

        ref_size = np.array(shape) * oversampling + 1
        assert epsf.data.shape == tuple(ref_size)

        # Verify basic EPSF properties
        assert len(fitted_stars) == 10
        assert epsf.data.sum() > 2  # Check it has reasonable total flux
        assert epsf.data.max() > 0.01  # Should have a peak

        # Check that the center region has higher values than edges
        center_y, center_x = np.array(ref_size) // 2
        center_val = epsf.data[center_y, center_x]
        edge_val = epsf.data[0, 0]
        assert center_val > edge_val  # Center should be brighter than edge

        # Test that residual computation works (basic functionality test)
        resid_star = fitted_stars[0].compute_residual_image(epsf)
        assert isinstance(resid_star, np.ndarray)
        assert resid_star.shape == fitted_stars[0].data.shape

    def test_epsf_fitting_bounds(self, epsf_test_data):
        size = 25
        oversampling = 4
        stars = extract_stars(epsf_test_data['nddata'],
                              epsf_test_data['init_stars'],
                              size=size)

        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=True, norm_radius=25,
                                   recentering_maxiters=5,
                                   fitter=EPSFFitter(fit_boxsize=31),
                                   smoothing_kernel='quadratic')

        # With a boxsize larger than the cutout we expect the fitting to
        # fail for all stars, due to star._fit_error_status
        match1 = 'The ePSF fitting failed for all stars'
        match2 = r'The star at .* cannot be fit because its fitting region '
        with (pytest.raises(ValueError, match=match1),
                pytest.warns(AstropyUserWarning, match=match2)):
            epsf_builder(stars)

    def test_resample_residual_masked_core(self):
        data = np.ones((7, 7), dtype=float)
        weights = np.ones_like(data)
        weights[2:5, 2:5] = 0.0
        star = EPSFStar(data, weights=weights, cutout_center=(3.0, 3.0))
        stars = EPSFStars([star])

        epsf_builder = EPSFBuilder(oversampling=2, maxiters=1,
                                   progress_bar=False)
        epsf = epsf_builder._create_initial_epsf(stars)

        # Regression test: masked pixels must not cause a boolean-index
        # mismatch in _resample_residual.
        resampled_img, img_weights, x_coords_img, y_coords_img = (
            epsf_builder._resample_residual(star, epsf)
        )

        assert resampled_img.shape == epsf.data.shape
        assert img_weights.shape == epsf.data.shape
        assert x_coords_img.shape == epsf.data.shape
        assert y_coords_img.shape == epsf.data.shape

        cy, cx = np.array(epsf.data.shape) // 2
        assert np.isnan(resampled_img[cy, cx])

    def test_epsf_build_invalid_fitter(self):
        """
        Test that the input fitter is an EPSFFitter instance.
        """
        match = 'fitter must be an EPSFFitter instance'
        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=EPSFFitter, maxiters=3)

        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=TRFLSQFitter(), maxiters=3)

        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=TRFLSQFitter, maxiters=3)


def test_epsfbuilder_inputs():
    # invalid inputs
    match = "'oversampling' must be specified"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=None)
    match = 'oversampling must be > 0'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=-1)
    match = 'maxiters must be a positive number'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(maxiters=-1)
    match = 'oversampling must be > 0'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=[-1, 4])
    match = 'flux_ppe_damping must be in the range \\[0, 1\\]'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(flux_ppe_damping=1.5)
    match = 'flux_ppe_update_every must be a positive integer'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(flux_ppe_update_every=0)
    match = 'residual_star_rms_clip must be positive or None'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_star_rms_clip=0)
    match = 'residual_outlier_clip must be positive or None'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_outlier_clip=-1)
    match = 'residual_min_valid_samples must be a positive integer'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_min_valid_samples=0)
    match = r'residual_update_fraction must be in the range \(0, 1\]'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_update_fraction=0.0)
    match = 'residual_despike_threshold must be a positive number'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_despike_threshold=0.0)
    match = 'residual_despike_passes must be a positive integer'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_despike_passes=0)
    match = "residual_despike_mode must be 'threshold' or 'strict'"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_despike_mode='invalid')

    # valid inputs
    EPSFBuilder(oversampling=6)
    EPSFBuilder(oversampling=[4, 6])
    EPSFBuilder(flux_ppe_damping=0.25, flux_ppe_update_every=3)
    EPSFBuilder(residual_star_rms_clip=None, residual_outlier_clip=3.0,
                residual_min_valid_samples=3, residual_update_fraction=0.25,
                residual_despike_boxsize=5, residual_despike_passes=3,
                residual_despike_mode='strict')

    # invalid inputs
    for sigma_clip in [None, [], 'a']:
        match = 'sigma_clip must be an astropy.stats.SigmaClip instance'
        with pytest.raises(TypeError, match=match):
            EPSFBuilder(sigma_clip=sigma_clip)

    # valid inputs
    EPSFBuilder(sigma_clip=SigmaClip(sigma=2.5, cenfunc='mean', maxiters=2))


def test_resample_epsf_anisotropic_grid():
    y_oversamp_in, x_oversamp_in = (4, 2)
    y_size_in, x_size_in = (33, 21)  # (native - 1) * oversampling + 1

    y0 = (y_size_in - 1) / 2.0
    x0 = (x_size_in - 1) / 2.0
    yy, xx = np.indices((y_size_in, x_size_in), dtype=float)
    x_img = (xx - x0) / x_oversamp_in
    y_img = (yy - y0) / y_oversamp_in

    # A linear surface makes coordinate-mapping errors easy to detect.
    data = x_img + (2.0 * y_img)
    input_epsf = ImagePSF(data=data, oversampling=(y_oversamp_in, x_oversamp_in))

    builder = EPSFBuilder(oversampling=(2, 3), maxiters=1, progress_bar=False)
    resampled = builder._resample_epsf(input_epsf, builder.oversampling)

    expected_shape = (17, 31)
    assert resampled.shape == expected_shape

    y0_out = (expected_shape[0] - 1) / 2.0
    x0_out = (expected_shape[1] - 1) / 2.0
    yy_out, xx_out = np.indices(expected_shape, dtype=float)
    x_img_out = (xx_out - x0_out) / builder.oversampling[1]
    y_img_out = (yy_out - y0_out) / builder.oversampling[0]
    expected = x_img_out + (2.0 * y_img_out)

    assert_allclose(resampled, expected, atol=1e-8)


def test_apply_ppe_corrections_policy():
    star = EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                    origin=(10, 20))
    stars = EPSFStars([star])
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          flux_ppe_damping=0.5, flux_ppe_update_every=2)

    ppemap = PPEMap((2, 2),
                    np.full((2, 2), 0.2),
                    np.full((2, 2), 0.25),
                    np.full((2, 2), -0.5))

    corrected_iter1 = builder._apply_ppe_corrections(stars, ppemap,
                                                     iteration=1)
    assert_allclose(corrected_iter1[0].flux, star.flux)
    assert_allclose(corrected_iter1[0].center, (11.75, 22.5))

    corrected_iter2 = builder._apply_ppe_corrections(stars, ppemap,
                                                     iteration=2)
    assert_allclose(corrected_iter2[0].flux, star.flux / 1.2 * 0.5
                    + star.flux * 0.5)
    assert_allclose(corrected_iter2[0].center, (11.75, 22.5))

    corrected_final = builder._apply_ppe_corrections(stars, ppemap,
                                                     final=True)
    assert_allclose(corrected_final[0].flux, star.flux / 1.2)
    assert_allclose(corrected_final[0].center, (11.75, 22.5))


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_ppemap_plot_maps():
    import matplotlib.pyplot as plt

    ppemap = PPEMap((2, 3),
                    np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6]]),
                    np.array([[0.0, 0.1, 0.0],
                              [-0.1, 0.0, 0.1]]),
                    np.array([[0.05, 0.0, -0.05],
                              [0.1, 0.0, -0.1]]))

    fig = ppemap.plot_maps()

    assert len(fig.axes) == 6
    assert [ax.get_title() for ax in fig.axes[:3]] == ['Flux PPE', 'X PPE',
                                                        'Y PPE']

    plt.close(fig)
