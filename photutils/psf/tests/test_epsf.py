# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf module.
"""

import itertools
import warnings

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import (InverseVariance, NDData, StdDevUncertainty,
                            VarianceUncertainty)
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
                                   progress_bar=False)
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
                                   progress_bar=True,
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


def test_select_residual_stars_zero_flux_no_runtime_warning():
    builder = EPSFBuilder(maxiters=1, progress_bar=False)

    data = np.zeros((7, 7), dtype=float)
    weights = np.ones_like(data)
    star = EPSFStar(data, weights=weights, cutout_center=(3.0, 3.0))
    stars = EPSFStars([star])
    epsf = ImagePSF(data=np.ones((7, 7), dtype=float), oversampling=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        selected = builder._select_residual_stars(stars, epsf)

    assert len(selected) == 1
    assert selected[0] is star
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


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
    match = "calibrate_ppe entries must be 'Flux' and/or 'Position'"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(calibrate_ppe=('Flux', 'Bad'))
    match = "constrain_stars entries must be 'Flux' and/or 'Position'"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(constrain_stars=('Flux', 'Bad'))
    match = 'residual_star_rms_clip must be positive or None'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_star_rms_clip=0)
    match = 'residual_outlier_clip must be positive or None'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_outlier_clip=-1)
    match = 'residual_min_valid_samples must be a positive integer'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(residual_min_valid_samples=0)
    match = "pixel_interpolation_method must be 'cubic' or 'nearest'"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(pixel_interpolation_method='invalid')
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(convergence_mode='model')
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(center_convergence_percentile=90.0)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(mask_background_pixels=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(recentering_func=lambda data, mask=None: (0.0, 0.0))
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(recentering_maxiters=5)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(recentering_boxsize=(5, 5))
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(norm_radius=10)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(epsf_nonnegative=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(apply_position_ppe=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(apply_flux_ppe=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(apply_final_flux_ppe=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(plot_ppe_diagnostics=False)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(sigma_clip=None)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(residual_update_fraction=0.25)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(residual_despike_threshold=3.0)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(residual_despike_boxsize=5)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(residual_despike_passes=3)
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        EPSFBuilder(residual_despike_mode='strict')

    # valid inputs
    builder = EPSFBuilder(oversampling=6)
    EPSFBuilder(oversampling=[4, 6])
    EPSFBuilder(recenter_epsf=False)
    EPSFBuilder(clip_negative=False)
    EPSFBuilder(calibrate_ppe=('Position',))
    EPSFBuilder(constrain_stars=('Position',))
    EPSFBuilder(plot_diagnostics=False)
    EPSFBuilder(residual_star_rms_clip=None, residual_outlier_clip=3.0,
                residual_min_valid_samples=3, residual_despike=False)
    EPSFBuilder(interpolate_missing_pixels=False)
    EPSFBuilder(pixel_interpolation_method='nearest')
    assert builder.interpolate_missing_pixels
    assert not hasattr(builder, 'convergence_mode')
    assert not hasattr(builder, 'center_convergence_percentile')
    assert not hasattr(builder, 'epsf_change_tolerance')
    assert not hasattr(builder, 'residual_change_tolerance')
    assert not hasattr(builder, 'convergence_stable_iters')
    assert not hasattr(builder, 'mask_background_pixels')
    assert not hasattr(builder, 'recentering_func')
    assert not hasattr(builder, 'recentering_maxiters')
    assert not hasattr(builder, 'recentering_boxsize')
    assert not hasattr(builder, '_norm_radius')
    assert not hasattr(builder, 'epsf_nonnegative')
    assert not hasattr(builder, 'apply_position_ppe')
    assert not hasattr(builder, 'apply_flux_ppe')
    assert not hasattr(builder, 'apply_final_flux_ppe')
    assert not hasattr(builder, 'plot_ppe_diagnostics')
    assert not hasattr(builder, 'flux_ppe_damping')
    assert not hasattr(builder, 'residual_update_fraction')
    assert not hasattr(builder, 'residual_despike_threshold')
    assert not hasattr(builder, 'residual_despike_boxsize')
    assert not hasattr(builder, 'residual_despike_passes')
    assert not hasattr(builder, 'residual_despike_mode')
    assert not hasattr(builder, 'flux_ppe_update_every')
    assert not hasattr(builder, '_sigma_clip')
    assert builder.recenter_epsf is True
    assert builder.clip_negative is True
    assert builder.calibrate_ppe == ('Flux', 'Position')
    assert builder.constrain_stars == ('Flux', 'Position')
    assert builder.plot_diagnostics is True


def test_epsfbuilder_interpolate_missing_residual_pixels():
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          pixel_interpolation_method='nearest')
    residuals = np.array([[1.0, 2.0, 3.0],
                          [4.0, np.nan, 6.0],
                          [7.0, 8.0, 9.0]])

    result = builder._interpolate_missing_residual_pixels(residuals)

    assert np.all(np.isfinite(result))
    assert result[1, 1] in residuals[np.isfinite(residuals)]
    mask = ~np.isnan(residuals)
    assert_allclose(result[mask], residuals[mask])


def test_epsfbuilder_missing_residual_pixels_can_remain_unchanged():
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          interpolate_missing_pixels=False)
    residuals = np.array([[1.0, np.nan],
                          [3.0, 4.0]])

    result = builder._interpolate_missing_residual_pixels(residuals)

    assert_allclose(result, [[1.0, 0.0], [3.0, 4.0]])


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
                          calibrate_ppe=('Flux', 'Position'))

    ppemap = PPEMap((2, 2),
                    np.full((2, 2), 0.2),
                    np.full((2, 2), 0.25),
                    np.full((2, 2), -0.5))

    corrected_iter1 = builder._apply_ppe_corrections(stars, ppemap,
                                                     iteration=1)
    assert_allclose(corrected_iter1[0].flux, star.flux / 1.2 * 0.8
                    + star.flux * 0.2)
    assert_allclose(corrected_iter1[0].center, (11.75, 22.5))

    corrected_iter2 = builder._apply_ppe_corrections(stars, ppemap,
                                                     iteration=2)
    assert_allclose(corrected_iter2[0].flux, star.flux / 1.2 * 0.8
                    + star.flux * 0.2)
    assert_allclose(corrected_iter2[0].center, (11.75, 22.5))

    corrected_final = builder._apply_ppe_corrections(stars, ppemap,
                                                     final=True)
    assert_allclose(corrected_final[0].flux, star.flux / 1.2)
    assert_allclose(corrected_final[0].center, (11.75, 22.5))

    flux_only_builder = EPSFBuilder(oversampling=2, maxiters=1,
                                    progress_bar=False,
                                    calibrate_ppe=('Flux',))
    corrected_flux_only = flux_only_builder._apply_ppe_corrections(
        stars, ppemap, final=True)
    assert_allclose(corrected_flux_only[0].flux, star.flux / 1.2)
    assert_allclose(corrected_flux_only[0].center, star.center)

    position_only_builder = EPSFBuilder(oversampling=2, maxiters=1,
                                        progress_bar=False,
                                        calibrate_ppe=('Position',))
    corrected_position_only = position_only_builder._apply_ppe_corrections(
        stars, ppemap, final=True)
    assert_allclose(corrected_position_only[0].flux, star.flux)
    assert_allclose(corrected_position_only[0].center, (11.75, 22.5))


def test_apply_linked_star_constraints_policy():
    class DummyStars:
        def __init__(self):
            self.calls = []

        def constrain_linked_centres(self):
            self.calls.append('Position')

        def constrain_linked_fluxes(self):
            self.calls.append('Flux')

    stars = DummyStars()
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          constrain_stars=('Position',))
    returned = builder._apply_linked_star_constraints(stars)
    assert returned is stars
    assert stars.calls == ['Position']

    stars = DummyStars()
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          constrain_stars=('Flux',))
    builder._apply_linked_star_constraints(stars)
    assert stars.calls == ['Flux']

    stars = DummyStars()
    builder = EPSFBuilder(oversampling=2, maxiters=1, progress_bar=False,
                          constrain_stars=('Flux', 'Position'))
    builder._apply_linked_star_constraints(stars)
    assert stars.calls == ['Position', 'Flux']


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
