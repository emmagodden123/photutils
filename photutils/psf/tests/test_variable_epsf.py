# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for experimental variable ePSF classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.psf.epsf_stars import EPSFStar, EPSFStars
from photutils.psf.variable_epsf import VariableEPSFModel
from photutils.psf.variable_epsf import VariableEPSFFitter


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


def test_variable_epsf_fitter_fwhm_only_no_exposure_time_warning():
    coeff_data = np.ones((1, 3, 3), dtype=float)
    fwhm_coeff_data = np.zeros((1, 3, 3), dtype=float)
    model = VariableEPSFModel(coeff_data, oversampling=1,
                              detector_shape=(10, 10), degree=0,
                              dependencies=('FWHM',),
                              fwhm_coeff_data=fwhm_coeff_data,
                              fwhm_degree=1, fwhm_reference=2.0,
                              fwhm_scale=1.0)
    star = EPSFStar(np.ones((3, 3), dtype=float), cutout_center=(1.0, 1.0),
                    fwhm=2.0)
    fitter = VariableEPSFFitter(fitter=_NoOpFitter(), fit_boxsize=None)

    fitted = fitter(model, EPSFStars([star]))

    assert fitted[0]._fit_error_status == 0
