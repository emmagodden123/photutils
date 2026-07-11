import numpy as np

from photutils.psf.spatial_epsf import SpatialEPSFBuilder


class _DummyStar:
    def __init__(self):
        self.center = (0.0, 0.0)
        self.flux = 1.0
        # mask has one masked pixel (middle); `_xidx_centered`/_yidx_centered
        # should list only the unmasked sample coordinates
        self.mask = np.array([False, True, False])
        self._xidx_centered = np.array([0.0, 0.0])
        self._yidx_centered = np.array([0.0, 0.0])

    def compute_residual_image(self, local_epsf):
        return np.array([0.1, 0.2, 0.3])


class _DummySpatialModel:
    def __init__(self):
        self.oversampling = (1, 1)
        self.shape = (2, 2)

    def make_image_psf(self, x, y):
        return None


def test_resample_residual_ignores_masked_pixels():
    builder = SpatialEPSFBuilder()
    star = _DummyStar()
    model = _DummySpatialModel()

    resampled, weights, x_coords, y_coords = builder._resample_residual(
        star, model)

    assert resampled.shape == model.shape
    assert weights.shape == model.shape
    assert x_coords.shape == model.shape
    assert y_coords.shape == model.shape