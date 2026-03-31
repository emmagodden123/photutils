# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for experimental spatial ePSF classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.psf.spatial_epsf import (SpatialEPSFBuilder, SpatialEPSFFitter,
                                        SpatialEPSFModel)
from photutils.psf.epsf_stars import EPSFStar, EPSFStars


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
                                 residual_min_valid_stars=1,
                                 update_trust_map=True,
                                 trust_map_floor=1.0e-6,
                                 apply_ppe_corrections=False)

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
