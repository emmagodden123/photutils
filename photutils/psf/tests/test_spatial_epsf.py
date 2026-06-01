# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for experimental spatial ePSF classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.centroids import centroid_com
from photutils.psf.spatial_epsf import (SpatialEPSFBuilder, SpatialEPSFFitter,
                                        SpatialEPSFModel)
from photutils.psf.epsf_stars import EPSFStar, EPSFStars
from photutils.psf.image_models import ImagePSF


class _ShiftFitter:
    def __init__(self, shifts):
        self.shifts = list(shifts)
        self.calls = 0
        self.fit_info = {'ierr': 1}
        self.received_weights = []

    def __call__(self, model, x, y, z, weights=None, **kwargs):
        self.received_weights.append(None if weights is None
                                     else np.array(weights, copy=True))
        shift = self.shifts[min(self.calls, len(self.shifts) - 1)]
        self.calls += 1

        model.x_0 = float(shift)
        model.y_0 = 0.0
        model.flux = float(model.flux.value)
        return model


class _ConstraintTracker:
    def __init__(self):
        self.center_calls = 0
        self.flux_calls = 0

    def __deepcopy__(self, memo):
        new = self.__class__()
        new.center_calls = self.center_calls
        new.flux_calls = self.flux_calls
        return new

    def constrain_linked_centres(self):
        self.center_calls += 1

    def constrain_linked_fluxes(self):
        self.flux_calls += 1


class _TestImagePSF(ImagePSF):
    pass


def _make_spatial_model(shape=(9, 9), detector_shape=(100, 100)):
    coeff_data = np.zeros((1, shape[0], shape[1]), dtype=float)
    coeff_data[0] = 1.0
    return SpatialEPSFModel(coeff_data, oversampling=1,
                            detector_shape=detector_shape, degree=0)


def test_spatial_epsf_fitter_applies_model_weight_map():
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    weights=np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))
    stars = EPSFStars([star])

    fitter_backend = _ShiftFitter(shifts=[0.0])

    def weight_map(local_epsf, star):
        return np.full(local_epsf.data.shape, 0.25)

    fitter = SpatialEPSFFitter(fitter=fitter_backend, fit_boxsize=None,
                               model_weight_map=weight_map,
                               model_weight_maxiters=1)
    fitted = fitter(_make_spatial_model(), stars)

    assert fitted[0]._fit_error_status == 0
    assert fitter_backend.calls == 1
    assert_allclose(fitter_backend.received_weights[0], 0.25)


def test_spatial_epsf_fitter_reweights_with_updated_center():
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    weights=np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))
    stars = EPSFStars([star])

    fitter_backend = _ShiftFitter(shifts=[0.4, 0.0])
    centers_seen = []

    def weight_map(local_epsf, star):
        centers_seen.append(tuple(star.cutout_center))
        return np.ones_like(local_epsf.data)

    fitter = SpatialEPSFFitter(fitter=fitter_backend, fit_boxsize=None,
                               model_weight_map=weight_map,
                               model_weight_maxiters=3,
                               model_weight_center_tol=1.0e-6)
    fitted = fitter(_make_spatial_model(), stars)

    assert fitter_backend.calls == 2
    assert centers_seen[0] == (2.0, 2.0)
    assert_allclose(centers_seen[1], (2.4, 2.0), atol=1.0e-12)
    assert_allclose(fitted[0].cutout_center, (2.4, 2.0), atol=1.0e-12)


def test_spatial_epsf_fitter_model_weight_map_shape_validation():
    star = EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0))
    stars = EPSFStars([star])
    fitter_backend = _ShiftFitter(shifts=[0.0])

    fitter = SpatialEPSFFitter(fitter=fitter_backend, fit_boxsize=None,
                               model_weight_map=np.ones((3, 3)))

    with pytest.raises(ValueError, match='model_weight_map must have'):
        fitter(_make_spatial_model(shape=(9, 9)), stars)


def test_spatial_epsf_model_trust_map_shape_validation():
    coeff_data = np.ones((1, 9, 9), dtype=float)
    with pytest.raises(ValueError, match='trust_map must have'):
        SpatialEPSFModel(coeff_data, oversampling=1,
                         detector_shape=(100, 100), degree=0,
                         trust_map=np.ones((5, 5), dtype=float))


def test_spatial_epsf_fitter_uses_model_trust_map():
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    weights=np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))
    stars = EPSFStars([star])

    trust_map = np.ones((9, 9), dtype=float)
    trust_map[:, :4] = 5.0
    spatial_model = _make_spatial_model(shape=(9, 9))
    spatial_model.trust_map = trust_map

    fitter_backend = _ShiftFitter(shifts=[0.0])
    fitter = SpatialEPSFFitter(fitter=fitter_backend, fit_boxsize=None,
                               model_weight_maxiters=1)
    fitter(spatial_model, stars)

    passed_weights = fitter_backend.received_weights[0]
    assert np.all(np.isfinite(passed_weights))
    assert np.nanmax(passed_weights) > np.nanmin(passed_weights)


def test_spatial_epsf_builder_trust_map_rms_from_residuals():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=0, maxiters=1,
                                 residual_min_valid_samples=1,
                                 calibrate_ppe=())

    residuals = np.array([
        [[1.0, 2.0], [3.0, np.nan]],
        [[3.0, 6.0], [4.0, np.nan]],
    ])
    weights = np.ones_like(residuals)

    trust_map = builder._compute_trust_map_from_residuals(residuals, weights)

    expected = np.array([
        [np.sqrt((1.0**2 + 3.0**2) / 2.0), np.sqrt((2.0**2 + 6.0**2) / 2.0)],
        [np.sqrt((3.0**2 + 4.0**2) / 2.0), np.nan],
    ])
    expected_fill = np.nanmedian(expected)
    expected[~np.isfinite(expected)] = expected_fill

    assert_allclose(trust_map, expected, atol=1.0e-12)


def test_spatial_epsf_builder_infers_detector_geometry_from_samples():
    stars = EPSFStars([
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 origin=(50, 100)),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 origin=(90, 140)),
    ])

    builder = SpatialEPSFBuilder(oversampling=1, degree=1, maxiters=1,
                                 residual_min_valid_samples=1,
                                 calibrate_ppe=())

    builder._resolve_detector_geometry(stars)

    assert_allclose(builder.detector_origin, (102.0, 52.0), atol=1.0e-12)
    assert_allclose(builder.detector_span, (40.0, 40.0), atol=1.0e-12)
    assert_allclose(builder.detector_shape, (41, 41), atol=1.0e-12)

    model = builder._create_initial_model(stars)
    xnorm, ynorm = model._normalize_detector_position(
        np.array((52.0, 92.0)), np.array((102.0, 142.0)))
    assert_allclose(xnorm, (-1.0, 1.0), atol=1.0e-12)
    assert_allclose(ynorm, (-1.0, 1.0), atol=1.0e-12)


def test_spatial_epsf_builder_recenter_epsf_toggle():
    y, x = np.indices((9, 9), dtype=float)
    coeff_data = np.exp(-((x - 5.0)**2 + (y - 3.5)**2) / (2.0 * 0.8**2))
    coeff_data = coeff_data[np.newaxis, :, :]
    sample_positions = [(50.0, 50.0)]

    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=0, recenter_epsf=True)
    recentered = builder._recenter_coefficients(coeff_data, sample_positions)

    y0, x0 = centroid_com(coeff_data[0])
    y1, x1 = centroid_com(recentered[0])
    target = np.array((4.0, 4.0))

    before = np.sum((np.array((y0, x0)) - target)**2)
    after = np.sum((np.array((y1, x1)) - target)**2)
    assert after < before

    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=0, recenter_epsf=False)
    unchanged = coeff_data.copy()
    if builder.recenter_epsf:
        unchanged = builder._recenter_coefficients(unchanged, sample_positions)
    assert_allclose(unchanged, coeff_data, atol=1.0e-12)


def test_spatial_epsf_builder_normalises_at_median_sample_position():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=1, normalise_epsf=True)
    builder.detector_origin = np.zeros(2, dtype=float)
    builder.detector_span = np.array((99.0, 99.0))

    coeff_data = np.zeros((3, 9, 9), dtype=float)
    coeff_data[0] = 2.0
    coeff_data[1] = 0.5

    sample_positions = [(10.0, 20.0), (50.0, 50.0), (90.0, 80.0)]
    normed = builder._normalise_coefficients(coeff_data.copy(),
                                             sample_positions)

    model = SpatialEPSFModel(normed, oversampling=1, detector_shape=(100, 100),
                             degree=1, normalize_local_epsf=False)
    local = model.local_epsf_data(50.0, 50.0)
    total = model._local_epsf_normalization(local)
    assert_allclose(total, 1.0, atol=1.0e-12)


def test_spatial_epsf_model_local_normalisation_toggle():
    coeff_data = np.zeros((1, 9, 9), dtype=float)
    coeff_data[0] = 3.0

    model = SpatialEPSFModel(coeff_data, oversampling=1,
                             detector_shape=(100, 100), degree=0,
                             normalize_local_epsf=True)
    total = model._local_epsf_normalization(model.make_image_psf(30.0,
                                                                 40.0).data)
    assert_allclose(total, 1.0, atol=1.0e-12)

    model = SpatialEPSFModel(coeff_data, oversampling=1,
                             detector_shape=(100, 100), degree=0,
                             normalize_local_epsf=False)
    total = model._local_epsf_normalization(model.make_image_psf(30.0,
                                                                 40.0).data)
    assert total > 1.0


def test_spatial_epsf_builder_auto_quadratic_offset_core_size():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=0)

    assert_allclose(builder._auto_quadratic_offset_core_size((9, 9)), (3, 3))
    assert_allclose(builder._auto_quadratic_offset_core_size((15, 21)),
                    (5, 7))
    assert_allclose(builder._auto_quadratic_offset_core_size((5, 7)), (3, 3))


def test_spatial_epsf_builder_interpolates_missing_coefficients():
    builder = SpatialEPSFBuilder(detector_shape=(10, 10),
                                 coefficient_interpolation_method='nearest')
    coeff = np.array([[[1.0, np.nan], [3.0, 4.0]]])

    result = builder._interpolate_missing_coefficient_images(coeff)

    assert np.all(np.isfinite(result))
    assert result[0, 0, 1] in (1.0, 3.0, 4.0)


def test_spatial_epsf_builder_residual_outlier_clip_validation():
    with pytest.raises(ValueError,
                       match='residual_outlier_clip must be positive or None'):
        SpatialEPSFBuilder(detector_shape=(100, 100),
                           residual_outlier_clip=-1)

    builder = SpatialEPSFBuilder(detector_shape=(100, 100),
                                 residual_outlier_clip=None)
    assert builder._sigma_clip is None


def test_spatial_epsf_builder_infers_ppe_settings_from_main_settings():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100),
                                 oversampling=(2, 3), degree=2,
                                 residual_min_valid_samples=7)
    assert builder.ppe_degree == 2
    assert_allclose(builder.ppe_supersampling, (2, 3))
    assert builder.ppe_min_valid_samples == 7


def test_spatial_epsf_builder_epsf_class_is_used():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100),
                                 epsf_class=_TestImagePSF)
    assert builder.epsf_class is _TestImagePSF

    model = SpatialEPSFModel(np.ones((1, 9, 9), dtype=float), oversampling=1,
                             detector_shape=(100, 100), degree=0,
                             epsf_class=_TestImagePSF)
    local = model.make_image_psf(30.0, 40.0)
    assert isinstance(local, _TestImagePSF)


def test_spatial_epsf_builder_residual_star_rms_clip_validation():
    with pytest.raises(ValueError,
                       match='residual_star_rms_clip must be positive or None'):
        SpatialEPSFBuilder(detector_shape=(100, 100),
                           residual_star_rms_clip=-1)


def test_spatial_epsf_builder_select_residual_stars_rms_clip(monkeypatch):
    stars = EPSFStars([
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='good0'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='good1'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='good2'),
        EPSFStar(np.ones((5, 5), dtype=float), cutout_center=(2.0, 2.0),
                 id_label='bad'),
    ])
    rms_by_id = {'good0': 0.00, 'good1': 0.05, 'good2': 0.08, 'bad': 0.50}

    def fake_compute_residual_image(self, local_epsf):
        return np.full(self.shape, rms_by_id[self.id_label] * self.flux,
                       dtype=float)

    monkeypatch.setattr(EPSFStar, 'compute_residual_image',
                        fake_compute_residual_image)

    builder = SpatialEPSFBuilder(detector_shape=(100, 100), oversampling=1,
                                 degree=0, residual_star_rms_clip=3.0)
    selected = builder._select_residual_stars(stars, _make_spatial_model())

    kept_ids = [star.id_label for star in selected.all_good_stars]
    assert kept_ids == ['good0', 'good1', 'good2']


def test_spatial_epsf_builder_calibrate_and_constrain_api():
    builder = SpatialEPSFBuilder(detector_shape=(100, 100),
                                 calibrate_ppe=('Position',),
                                 constrain_stars=('Flux',))
    assert builder.calibrate_ppe == ('Position',)
    assert builder.constrain_stars == ('Flux',)

    constrained = builder._apply_linked_constraints(_ConstraintTracker())
    assert constrained.center_calls == 0
    assert constrained.flux_calls == 1

    builder = SpatialEPSFBuilder(detector_shape=(100, 100),
                                 calibrate_ppe=(),
                                 constrain_stars=('Position',))
    assert builder._fit_spatial_ppe_model(EPSFStars([])).flux_coeff.shape[0] >= 1

    with pytest.raises(ValueError,
                       match="calibrate_ppe entries must be 'Flux' and/or "
                             "'Position'"):
        SpatialEPSFBuilder(detector_shape=(100, 100),
                           calibrate_ppe=('Flux', 'Bad'))

    with pytest.raises(ValueError,
                       match="constrain_stars entries must be 'Flux' and/or "
                             "'Position'"):
        SpatialEPSFBuilder(detector_shape=(100, 100),
                           constrain_stars=('Flux', 'Bad'))


def test_spatial_epsf_fitter_backward_compatible_without_trust_map_attr():
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    weights=np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))
    stars = EPSFStars([star])

    spatial_model = _make_spatial_model(shape=(9, 9))
    delattr(spatial_model, 'trust_map')

    fitter_backend = _ShiftFitter(shifts=[0.0])
    fitter = SpatialEPSFFitter(fitter=fitter_backend, fit_boxsize=None,
                               model_weight_maxiters=2)

    fitted = fitter(spatial_model, stars)

    assert fitter_backend.calls == 1
    assert fitted[0]._fit_error_status == 0


def test_spatial_epsf_fitter_accepts_ndarray_fit_boxsize():
    star = EPSFStar(np.ones((9, 9), dtype=float),
                    weights=np.ones((9, 9), dtype=float),
                    cutout_center=(4.0, 4.0))
    stars = EPSFStars([star])

    fitter_backend = _ShiftFitter(shifts=[0.0])
    fitter = SpatialEPSFFitter(fitter=fitter_backend,
                               fit_boxsize=np.array([5, 5]),
                               model_weight_maxiters=1)

    fitted = fitter(_make_spatial_model(), stars)

    assert fitter_backend.calls == 1
    assert fitted[0]._fit_error_status == 0
