# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the gridded ePSF builder.
"""

import numpy as np
from numpy.testing import assert_allclose

from photutils.psf.epsf_stars import EPSFStar, LinkedEPSFStar
from photutils.psf.gridded_epsf import GriddedEPSFBuilder
from photutils.psf.image_models import ImagePSF


class SimpleWCS:
    def pixel_to_world_values(self, x, y):
        return x, y

    def world_to_pixel_values(self, lon, lat):
        return lon, lat


class DummyBuilder:
    def __init__(self):
        self.maxiters = 5
        self.calls = []

    def build_epsf(self, stars, *, init_model=None):
        self.calls.append((self.maxiters, stars, init_model))
        return 'epsf', stars


def make_star(x, y):
    data = np.ones((5, 5), dtype=float)
    return EPSFStar(data, cutout_center=(2, 2), origin=(x - 2, y - 2),
                    wcs_large=SimpleWCS())


def test_partition_preserves_linked_stars():
    builder = GriddedEPSFBuilder((100, 100), (1, 2), 2, maxiters=1,
                                 min_stars_per_gridcell=1)
    linked = LinkedEPSFStar([make_star(20, 50), make_star(30, 50)])

    grid_stars = builder._partition_stars_by_gridcell([linked])

    assert isinstance(grid_stars[0][0][0], LinkedEPSFStar)
    assert grid_stars[0][0].n_all_stars == 2
    assert grid_stars[0][1] is None


def test_build_grid_cell_epsf_can_override_maxiters_temporarily():
    dummy_builder = DummyBuilder()

    epsf, stars = GriddedEPSFBuilder._build_grid_cell_epsf(
        dummy_builder, 'stars', init_model='init', maxiters=1)

    assert epsf == 'epsf'
    assert stars == 'stars'
    assert dummy_builder.calls == [(1, 'stars', 'init')]
    assert dummy_builder.maxiters == 5


def test_make_gridded_psf_model_fills_missing_cells_from_nearest_valid():
    builder = GriddedEPSFBuilder((100, 100), (2, 2), 2, maxiters=1,
                                 min_stars_per_gridcell=1)

    epsf1 = ImagePSF(np.full((5, 5), 1.0), oversampling=2)
    epsf2 = ImagePSF(np.full((5, 5), 2.0), oversampling=2)
    epsf3 = ImagePSF(np.full((5, 5), 3.0), oversampling=2)

    model = builder._make_gridded_psf_model([None, epsf1, epsf2, epsf3])

    assert_allclose(model.data[0], epsf1.data)
    assert_allclose(model.data[1], epsf1.data)
    assert_allclose(model.data[2], epsf2.data)
    assert_allclose(model.data[3], epsf3.data)
