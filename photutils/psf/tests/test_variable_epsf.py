# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for experimental variable ePSF classes.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf.epsf_stars import EPSFStar, EPSFStars
from photutils.psf.variable_epsf import VariableEPSFBuilder
from photutils.psf.variable_epsf import VariableEPSFFitter
from photutils.psf.variable_epsf import VariableEPSFModel


def test_variable_epsf_model_fwhm_dependency():
    coeff_data = np.ones((1, 3, 3), dtype=float)
    fwhm_coeff_data = np.full((1, 3, 3), 2.0, dtype=float)

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('Position', 'FWHM'),
                              fwhm_coeff_data=fwhm_coeff_data,
                              fwhm_degree=1, fwhm_reference=2.0,
                              fwhm_scale=2.0,
                              normalize_local_epsf=False)

    assert model.has_fwhm_dependency
    assert_allclose(model.local_epsf_data(5.0, 5.0, fwhm=2.0), 1.0)
    assert_allclose(model.local_epsf_data(5.0, 5.0, fwhm=4.0), 3.0)


def test_variable_epsf_model_additive_fwhm_dependency():
    coeff_data = np.full((1, 3, 3), 2.0, dtype=float)
    fwhm_coeff_data = np.full((1, 3, 3), 2.0, dtype=float)

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('FWHM',),
                              fwhm_correction_mode='additive',
                              fwhm_coeff_data=fwhm_coeff_data,
                              fwhm_degree=1, fwhm_reference=2.0,
                              fwhm_scale=2.0,
                              normalize_local_epsf=False)

    assert_allclose(model.local_epsf_data(5.0, 5.0, fwhm=2.0), 2.0)
    assert_allclose(model.local_epsf_data(5.0, 5.0, fwhm=4.0), 4.0)


def test_variable_epsf_model_additive_flux_dependency():
    coeff_data = np.full((1, 3, 3), 2.0, dtype=float)
    flux_coeff_data = np.full((1, 3, 3), 2.0, dtype=float)

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('Flux',),
                              flux_correction_mode='additive',
                              flux_coeff_data=flux_coeff_data,
                              flux_degree=1, flux_reference=10.0,
                              flux_scale=10.0,
                              normalize_local_epsf=False)

    assert_allclose(model.local_epsf_data(5.0, 5.0, flux=10.0), 2.0)
    assert_allclose(model.local_epsf_data(5.0, 5.0, flux=20.0), 4.0)


def test_variable_epsf_model_constant_flux_dependency():
    coeff_data = np.full((1, 3, 3), 2.0, dtype=float)
    flux_coeff_data = np.full((1, 3, 3), 0.5, dtype=float)

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('Flux',),
                              flux_coeff_data=flux_coeff_data,
                              flux_degree=0, normalize_local_epsf=False)

    assert_allclose(model.local_epsf_data(5.0, 5.0, flux=1.0), 3.0)
    assert_allclose(model.local_epsf_data(5.0, 5.0, flux=100.0), 3.0)


def test_variable_epsf_model_invalid_correction_mode():
    coeff_data = np.ones((1, 3, 3), dtype=float)

    with pytest.raises(ValueError,
                       match="flux_correction_mode must be 'multiplicative'"):
        VariableEPSFModel(coeff_data, oversampling=1,
                          detector_shape=(10, 10), degree=0,
                          flux_correction_mode='bad')


def test_variable_epsf_model_spatial_coeff_data_alias():
    spatial_coeff_data = np.ones((1, 3, 3), dtype=float)

    model = VariableEPSFModel(spatial_coeff_data=spatial_coeff_data,
                              oversampling=1, detector_shape=(10, 10),
                              degree=0)
    legacy_model = VariableEPSFModel(coeff_data=spatial_coeff_data,
                                     oversampling=1,
                                     detector_shape=(10, 10), degree=0)

    assert_allclose(model.spatial_coeff_data, spatial_coeff_data)
    assert_allclose(legacy_model.spatial_coeff_data, spatial_coeff_data)
    assert_allclose(model.coeff_data, model.spatial_coeff_data)


def test_variable_epsf_model_clip_negative():
    coeff_data = np.ones((1, 5, 5), dtype=float)
    coeff_data[0, 0, 0] = -0.05

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              clip_negative=True,
                              normalize_local_epsf=False)

    data = model.local_epsf_data(5.0, 5.0)

    assert data[0, 0] == 0.0
    assert_allclose(data[2, 2], 1.0)


@pytest.mark.parametrize(
    'dependencies',
    [
        ('Position',),
        ('Flux',),
        ('FWHM',),
        ('Position', 'Flux'),
        ('Position', 'FWHM'),
        ('Flux', 'FWHM'),
        ('Position', 'Flux', 'FWHM'),
    ],
)
def test_variable_epsf_model_dependency_combinations(dependencies):
    degree = 1 if 'Position' in dependencies else 0
    coeff_data = np.ones((3 if degree == 1 else 1, 3, 3), dtype=float)
    kwargs = {}
    if 'Flux' in dependencies:
        kwargs['flux_coeff_data'] = np.zeros((1, 3, 3), dtype=float)
    if 'FWHM' in dependencies:
        kwargs['fwhm_coeff_data'] = np.zeros((1, 3, 3), dtype=float)

    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=degree,
                              dependencies=dependencies,
                              normalize_local_epsf=False, **kwargs)

    data = model.local_epsf_data(5.0, 5.0, flux=1.0, fwhm=1.0)

    assert data.shape == (3, 3)


class _NoOpFitter:
    fit_info = {'ierr': 1}

    def __call__(self, model, x, y, z, weights=None, **kwargs):
        model.x_0 = 0.0
        model.y_0 = 0.0
        model.flux = float(model.flux.value)
        return model


class _LargeShiftFitter:
    fit_info = {'ierr': 1}

    def __call__(self, model, x, y, z, weights=None, **kwargs):
        model.x_0 = 2.0
        model.y_0 = 0.0
        model.flux = float(model.flux.value)
        return model


class _WarnButIerrOkFitter:
    fit_info = {'ierr': 1}

    def __call__(self, model, x, y, z, weights=None, **kwargs):
        model.x_0 = 0.4
        model.y_0 = 0.0
        model.flux = float(model.flux.value)
        import warnings
        warnings.warn('The fit may be unsuccessful; check: '\
                      'The maximum number of function evaluations is '
                      'exceeded.', AstropyUserWarning)
        return model


def test_variable_epsf_fitter_fwhm_only_no_exposure_time_warning():
    coeff_data = np.ones((1, 5, 5), dtype=float)
    fwhm_coeff_data = np.zeros((1, 5, 5), dtype=float)
    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('FWHM',),
                              fwhm_coeff_data=fwhm_coeff_data,
                              fwhm_degree=1, fwhm_reference=2.0,
                              fwhm_scale=1.0)
    star = EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                    fwhm=2.0)
    fitter = VariableEPSFFitter(fitter=_NoOpFitter(), fit_boxsize=None)

    fitted = fitter(model, EPSFStars([star]))

    assert fitted[0]._fit_error_status == 0


def test_variable_epsf_fitter_warns_once_for_overlap_failures():
    coeff_data = np.ones((1, 5, 5), dtype=float)
    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0)
    stars = EPSFStars([
        EPSFStar(np.ones((3, 3), dtype=float), cutout_center=(1.0, 1.0)),
        EPSFStar(np.ones((3, 3), dtype=float), cutout_center=(1.0, 1.0)),
    ])
    fitter = VariableEPSFFitter(fitter=_NoOpFitter(), fit_boxsize=5)

    with pytest.warns(AstropyUserWarning,
                      match='2 star\\(s\\) could not be fit') as warning_info:
        fitted = fitter(model, stars)

    assert len(warning_info) == 1
    assert_allclose([star._fit_error_status for star in fitted], [1, 1])


def test_variable_epsf_fitter_rejects_large_center_shifts():
    coeff_data = np.ones((1, 7, 7), dtype=float)
    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0)
    star = EPSFStar(np.ones((7, 7), dtype=float),
                    weights=np.ones((7, 7), dtype=float),
                    cutout_center=(3.0, 3.0))
    fitter = VariableEPSFFitter(fitter=_LargeShiftFitter(), fit_boxsize=3)

    with pytest.warns(AstropyUserWarning,
                      match='may not have been fit successfully'):
        fitted = fitter(model, EPSFStars([star]))

    assert fitted[0]._fit_error_status == 2
    assert_allclose(fitted[0].cutout_center, (3.0, 3.0), atol=1.0e-12)


def test_variable_epsf_fitter_warning_pattern_marks_failure():
    coeff_data = np.ones((1, 5, 5), dtype=float)
    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0)
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    weights=np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))
    fitter = VariableEPSFFitter(fitter=_WarnButIerrOkFitter(), fit_boxsize=None)

    with pytest.warns(AstropyUserWarning,
                      match='may not have been fit successfully'):
        fitted = fitter(model, EPSFStars([star]))

    assert fitted[0]._fit_error_status == 2
    assert_allclose(fitted[0].cutout_center, (2.0, 2.0), atol=1.0e-12)


def test_variable_epsf_builder_fractional_residuals_core_floor():
    residuals = np.ones((1, 3, 3), dtype=float)
    model_data = np.array([[
        [1.0e-8, 1.0e-5, 1.0e-3],
        [1.0e-2, 1.0, 1.0e-2],
        [1.0e-3, 1.0e-5, 1.0e-8],
    ]])

    frac = VariableEPSFBuilder._fractional_residuals(residuals, model_data)

    assert np.isnan(frac[0, 0, 0])
    assert np.isnan(frac[0, 0, 1])
    assert_allclose(frac[0, 1, 1], 1.0)


def test_variable_epsf_builder_correction_residual_modes():
    residuals = np.ones((1, 3, 3), dtype=float)
    model_data = np.zeros((1, 3, 3), dtype=float)
    model_data[0, 1, 1] = 2.0

    additive = VariableEPSFBuilder._correction_residuals(
        residuals, model_data, 'additive')
    multiplicative = VariableEPSFBuilder._correction_residuals(
        residuals, model_data, 'multiplicative')

    assert_allclose(additive, residuals)
    assert_allclose(multiplicative[0, 1, 1], 0.5)
    assert np.isnan(multiplicative[0, 0, 0])


def test_variable_epsf_builder_warns_for_sparse_flux_sections():
    builder = VariableEPSFBuilder(
        dependencies=('Flux',), fit_dependencies=('Flux',),
        detector_shape=(10, 10), flux_min_valid_samples=2,
        coefficient_interpolation_method='nearest')
    residuals = np.ones((3, 2, 2), dtype=float)
    residuals[1:, 0, 0] = np.nan
    weights = np.ones_like(residuals)
    effective_flux = np.array([1.0, 2.0, 3.0])

    with pytest.warns(AstropyUserWarning,
                      match='Insufficient sources') as warning_info:
        coeff, _, _ = builder._fit_flux_coefficients(
            residuals, weights, effective_flux)

    messages = [str(warning.message) for warning in warning_info]
    assert any('flux coefficient grid section' in message
               for message in messages)
    assert np.all(np.isfinite(coeff))
    assert_allclose(coeff[:, 0, 0], 0.0)


def test_variable_epsf_builder_fits_constant_flux_coefficients():
    builder = VariableEPSFBuilder(
        dependencies=('Flux',), fit_dependencies=('Flux',),
        detector_shape=(10, 10), flux_degree=0, flux_min_valid_samples=1,
        use_time_integrated_flux=False)
    residuals = np.full((3, 2, 2), 0.25, dtype=float)
    weights = np.ones_like(residuals)
    effective_flux = np.array([1.0, 10.0, 100.0])

    coeff, _, _ = builder._fit_flux_coefficients(
        residuals, weights, effective_flux)

    assert coeff.shape == (1, 2, 2)
    assert_allclose(coeff, 0.25)


def test_variable_epsf_builder_select_residual_stars_keeps_empty_selection(monkeypatch):
    stars = EPSFStars([
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='bad0'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='bad1'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='bad2'),
    ])

    def fake_compute_residual_image(self, local_epsf):
        return np.full(self.shape, 10.0 * self.flux, dtype=float)

    monkeypatch.setattr(EPSFStar, 'compute_residual_image',
                        fake_compute_residual_image)

    builder = VariableEPSFBuilder(dependencies=('Position',),
                                  fit_dependencies=('Position',),
                                  detector_shape=(10, 10), oversampling=1,
                                  degree=0, residual_star_rms_clip=1.0)
    monkeypatch.setattr(builder, '_mad_std', lambda data: -1.0)
    selected = builder._select_residual_stars(stars, VariableEPSFModel(
        np.ones((1, 5, 5), dtype=float), oversampling=1,
        detector_shape=(10, 10), degree=0,
        normalize_local_epsf=False))

    assert selected.n_good_stars == 0
    assert [star.id_label for star in selected.all_good_stars] == []
    assert all(star._excluded_from_fit for star in selected.all_stars)


def test_variable_epsf_builder_select_residual_stars_excludes_fit_failures():
    stars = EPSFStars([
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='ok0'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='fail'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='ok1'),
    ])
    stars.all_stars[1]._fit_error_status = 2

    builder = VariableEPSFBuilder(dependencies=('Position',),
                                  fit_dependencies=('Position',),
                                  detector_shape=(10, 10), oversampling=1,
                                  degree=0, residual_star_rms_clip=None)
    selected = builder._select_residual_stars(stars, VariableEPSFModel(
        np.ones((1, 5, 5), dtype=float), oversampling=1,
        detector_shape=(10, 10), degree=0,
        normalize_local_epsf=False))

    kept_ids = [star.id_label for star in selected.all_good_stars]
    assert kept_ids == ['ok0', 'ok1']
    assert selected.all_stars[1]._excluded_from_fit


def test_variable_epsf_builder_fit_dependencies_validation():
    with pytest.raises(ValueError, match='fit_dependencies entries'):
        VariableEPSFBuilder(dependencies=('Flux',),
                            fit_dependencies=('FWHM',))


def test_variable_epsf_builder_extends_init_model_dependencies():
    coeff_data = np.ones((1, 3, 3), dtype=float)
    fwhm_coeff_data = np.full((1, 3, 3), 0.25, dtype=float)
    init_model = VariableEPSFModel(
        coeff_data, oversampling=1, detector_shape=(10, 10), degree=0,
        dependencies=('Position', 'FWHM'),
        fwhm_coeff_data=fwhm_coeff_data, fwhm_degree=1,
        fwhm_reference=2.0, fwhm_scale=0.5,
        normalize_local_epsf=False)

    builder = VariableEPSFBuilder(
        dependencies=('Flux',), fit_dependencies=('Flux',),
        oversampling=1, detector_shape=(10, 10), degree=0)
    model = builder._create_initial_variable_model(
        EPSFStars([]), init_model=init_model)

    assert model.dependencies == ('Position', 'FWHM', 'Flux')
    assert_allclose(model.spatial_coeff_data, init_model.spatial_coeff_data)
    assert_allclose(model.fwhm_coeff_data, fwhm_coeff_data)
    assert_allclose(model.flux_coeff_data, 0.0)
    assert model.fwhm_reference == init_model.fwhm_reference
    assert model.fwhm_scale == init_model.fwhm_scale


def test_variable_epsf_builder_flux_only_preserves_frozen_terms(monkeypatch):
    spatial_coeff_data = np.ones((1, 3, 3), dtype=float)
    fwhm_coeff_data = np.full((1, 3, 3), 0.2, dtype=float)
    init_model = VariableEPSFModel(
        spatial_coeff_data, oversampling=1, detector_shape=(10, 10),
        degree=0, dependencies=('Position', 'FWHM'),
        fwhm_coeff_data=fwhm_coeff_data, fwhm_degree=1,
        fwhm_reference=2.0, fwhm_scale=1.0,
        normalize_local_epsf=False)
    star1 = EPSFStar(np.ones((3, 3), dtype=float),
                     cutout_center=(1.0, 1.0), origin=(0, 0), fwhm=2.0)
    star1.flux = 10.0
    star2 = EPSFStar(np.ones((3, 3), dtype=float),
                     cutout_center=(1.0, 1.0), origin=(0, 0), fwhm=2.0)
    star2.flux = 20.0
    stars = EPSFStars([star1, star2])
    builder = VariableEPSFBuilder(
        dependencies=('Flux',), fit_dependencies=('Flux',), oversampling=1,
        detector_shape=(10, 10), degree=0, maxiters=1,
        flux_min_valid_samples=1, normalise_epsf=True, recenter_epsf=True,
        use_time_integrated_flux=False)

    def fail_spatial_fit(*args, **kwargs):
        raise AssertionError('spatial coefficients should be frozen')

    residuals = np.ones((2, 3, 3), dtype=float) * 0.1
    weights = np.ones_like(residuals)
    coords = np.zeros_like(residuals)
    det_x = np.array([1.0, 2.0])
    det_y = np.array([1.0, 2.0])
    group_id = np.array([0, 1])

    monkeypatch.setattr(builder, '_fit_residual_coefficients',
                        fail_spatial_fit)
    monkeypatch.setattr(builder, '_fit_spatial_ppe_model',
                        lambda fitted_stars: None)
    monkeypatch.setattr(builder, '_apply_spatial_ppe_corrections',
                        lambda fitted_stars, ppe_model: fitted_stars)
    monkeypatch.setattr(builder, '_apply_linked_constraints',
                        lambda fitted_stars: fitted_stars)
    monkeypatch.setattr(builder, '_select_residual_stars',
                        lambda fitted_stars, model: fitted_stars)
    monkeypatch.setattr(builder, '_resample_residuals',
                        lambda fitted_stars, model: (residuals, weights,
                                                     coords, coords, det_x,
                                                     det_y, group_id))
    monkeypatch.setattr(builder, '_compute_trust_map_from_residuals',
                        lambda residuals, weights: np.ones((3, 3),
                                                           dtype=float))
    monkeypatch.setattr(builder, '_plot_iteration_diagnostics',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, '_plot_central_residual_fit',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, '_plot_residual_scatter_diagnostics',
                        lambda *args, **kwargs: None)
    builder.variable_fitter = lambda model, stars: stars

    model, _ = builder.build_epsf(stars, init_model=init_model)

    assert model.has_flux_dependency
    assert_allclose(model.spatial_coeff_data, spatial_coeff_data)
    assert_allclose(model.fwhm_coeff_data, fwhm_coeff_data)
    assert np.any(model.flux_coeff_data != 0.0)
