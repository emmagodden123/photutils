# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Experimental spatially varying ePSF classes.

This module implements a detector-position-dependent ePSF model where
each oversampled ePSF grid point is represented by a low-order
polynomial function of detector position. The intent is to provide an
alternative to patchwise/local ePSF building when the ePSF still varies
appreciably across the image.
"""

import copy
import warnings
from functools import partial

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve, map_coordinates
from scipy.spatial import QhullError

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround
from photutils.utils._stats import nanmedian
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['SpatialEPSFModel', 'SpatialPPEMapModel', 'SpatialEPSFFitter',
           'SpatialEPSFBuilder', 'PPEMap']


class SpatialEPSFModel:
    """
    A detector-position-dependent ePSF model.

    Parameters
    ----------
    coeff_data : 3D `~numpy.ndarray`
        Coefficient array with shape ``(n_basis, ny, nx)``.

    oversampling : int or tuple of int
        The integer oversampling factor(s) of the ePSF relative to the
        input stars.

    detector_shape : tuple of int
        The detector/image shape in ``(ny, nx)`` order.

    detector_origin : tuple of float, optional
        The detector-coordinate origin, in ``(y, x)`` order, used when
        normalizing detector positions. The default is ``(0, 0)``.

    detector_span : tuple of float, optional
        The detector-coordinate span, in ``(y, x)`` order, used when
        normalizing detector positions. By default this is inferred from
        ``detector_shape - 1``.

    degree : int, optional
        Polynomial degree in detector position. Supported values are 0,
        1, and 2. The default is 1.

    origin, fill_value : optional
        Passed through to local `ImagePSF` models.
    """

    def __init__(self, coeff_data, *, oversampling, detector_shape,
                 degree=1, origin=None, fill_value=0.0,
                 detector_origin=None, detector_span=None,
                 normalize_local_epsf=True, trust_map=None,
                 epsf_class=ImagePSF):
        self.coeff_data = np.asanyarray(coeff_data, dtype=float)
        if self.coeff_data.ndim != 3:
            raise ValueError('coeff_data must be a 3D array')

        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        self.detector_shape = as_pair('detector_shape', detector_shape,
                                      lower_bound=(2, 2))
        if detector_origin is None:
            detector_origin = (0.0, 0.0)
        self.detector_origin = np.asanyarray(detector_origin, dtype=float)
        if self.detector_origin.shape != (2,):
            raise ValueError('detector_origin must have shape (2,)')
        if not np.all(np.isfinite(self.detector_origin)):
            raise ValueError('detector_origin must contain only finite values')

        if detector_span is None:
            detector_span = self.detector_shape - 1
        self.detector_span = np.asanyarray(detector_span, dtype=float)
        if self.detector_span.shape != (2,):
            raise ValueError('detector_span must have shape (2,)')
        if not np.all(np.isfinite(self.detector_span)):
            raise ValueError('detector_span must contain only finite values')
        self.degree = int(degree)
        if self.degree not in (0, 1, 2):
            raise ValueError('degree must be 0, 1, or 2')

        expected_basis = len(self._basis_labels())
        if self.coeff_data.shape[0] != expected_basis:
            raise ValueError('coeff_data first axis does not match degree')

        self.origin = origin
        self.fill_value = fill_value
        self.normalize_local_epsf = bool(normalize_local_epsf)
        self.epsf_class = epsf_class
        if isinstance(self.epsf_class, partial):
            candidate = self.epsf_class.func
        else:
            candidate = self.epsf_class
        if not issubclass(candidate, ImagePSF):
            raise TypeError('epsf_class must be a subclass of ImagePSF')

        if trust_map is not None:
            trust_map = np.asanyarray(trust_map, dtype=float)
            if trust_map.shape != self.coeff_data.shape[1:]:
                raise ValueError('trust_map must have the same shape as the '
                                 'spatial ePSF image grid')
        self.trust_map = trust_map

    @property
    def shape(self):
        return self.coeff_data.shape[1:]

    @staticmethod
    def _basis_labels_for_degree(degree):
        if degree == 0:
            return ('1',)
        if degree == 1:
            return ('1', 'x', 'y')
        return ('1', 'x', 'y', 'x2', 'xy', 'y2')

    def _basis_labels(self):
        return self._basis_labels_for_degree(self.degree)

    def _normalize_detector_position(self, x, y):
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)

        xnorm = 2.0 * ((x - self.detector_origin[1])
                       / max(self.detector_span[1], 1.0)) - 1.0
        ynorm = 2.0 * ((y - self.detector_origin[0])
                       / max(self.detector_span[0], 1.0)) - 1.0
        return xnorm, ynorm

    def basis_vector(self, x, y):
        """
        Return the detector-position basis vector at ``(x, y)``.
        """
        xnorm, ynorm = self._normalize_detector_position(x, y)
        if self.degree == 0:
            return np.stack((np.ones_like(xnorm),), axis=0)
        if self.degree == 1:
            return np.stack((np.ones_like(xnorm), xnorm, ynorm), axis=0)
        return np.stack((np.ones_like(xnorm), xnorm, ynorm, xnorm**2,
                         xnorm * ynorm, ynorm**2), axis=0)

    def local_epsf_data(self, x, y):
        """
        Evaluate the local oversampled ePSF image at detector position
        ``(x, y)``.
        """
        basis = self.basis_vector(x, y)
        return np.tensordot(basis, self.coeff_data, axes=(0, 0))

    def make_image_psf(self, x, y):
        """
        Create a local `ImagePSF` for detector position ``(x, y)``.
        """
        data = self.local_epsf_data(x, y)
        if self.normalize_local_epsf:
            data = self._normalise_local_epsf_data(data)
        image_psf = self.epsf_class(data=data, oversampling=self.oversampling,
                                    origin=self.origin,
                                    fill_value=self.fill_value)
        trust_map = getattr(self, 'trust_map', None)
        if trust_map is not None:
            image_psf.trust_map = np.array(trust_map, copy=True)
        return image_psf

    def plot_model(self, x, y, *, plot_type='2d', grid_factor=1, ax=None,
                   cmap='viridis'):
        """
        Plot the local ePSF at detector position ``(x, y)``.

        Parameters
        ----------
        x, y : float
            Detector position where the local ePSF should be evaluated.

        plot_type : {'2d', '3d'}, optional
            The type of plot to generate. The default is ``'2d'``.

        grid_factor : int, optional
            The factor by which to refine the evaluation grid relative to
            the native oversampled ePSF grid. A value of 1 evaluates the
            model on its native oversampled grid. Larger values evaluate
            the model on a finer grid. The default is 1.

        ax : `~matplotlib.axes.Axes`, optional
            An existing matplotlib axes to plot on. For ``plot_type='3d'``,
            this must be a 3D axes.

        cmap : str, optional
            The matplotlib colormap to use. The default is ``'viridis'``.

        Returns
        -------
        fig, ax : tuple
            The matplotlib figure and axes containing the plot.
        """
        if plot_type not in ('2d', '3d'):
            raise ValueError("plot_type must be '2d' or '3d'")

        grid_factor = int(grid_factor)
        if grid_factor <= 0:
            raise ValueError('grid_factor must be a positive integer')

        import matplotlib.pyplot as plt

        local_epsf = self.make_image_psf(x, y)
        half_x = (self.shape[1] - 1) / (2.0 * self.oversampling[1])
        half_y = (self.shape[0] - 1) / (2.0 * self.oversampling[0])

        nx = self.shape[1] * grid_factor
        ny = self.shape[0] * grid_factor
        xgrid = np.linspace(-half_x, half_x, nx)
        ygrid = np.linspace(-half_y, half_y, ny)
        yy, xx = np.meshgrid(ygrid, xgrid, indexing='ij')
        values = local_epsf.evaluate(x=xx, y=yy, flux=1.0, x_0=0.0, y_0=0.0)

        fig = None
        if ax is None:
            if plot_type == '3d':
                fig = plt.figure(figsize=(7, 5.5))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = ax.figure

        if plot_type == '2d':
            im = ax.imshow(values, origin='lower',
                           extent=(-half_x, half_x, -half_y, half_y),
                           cmap=cmap, aspect='equal')
            ax.set_xlabel('Model x (pixels)')
            ax.set_ylabel('Model y (pixels)')
            fig.colorbar(im, ax=ax, label='ePSF value')
        else:
            surf = ax.plot_surface(xx, yy, values, cmap=cmap,
                                   linewidth=0, antialiased=True)
            ax.set_xlabel('Model x (pixels)')
            ax.set_ylabel('Model y (pixels)')
            ax.set_zlabel('ePSF value')
            fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1,
                         label='ePSF value')

        ax.set_title(f'Local ePSF at detector position ({x:.2f}, {y:.2f})')
        return fig, ax

    def plot_trust_map(self, *, ax=None, cmap='viridis'):
        """
        Plot the global trust map on the oversampled ePSF grid.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            An existing matplotlib axes to plot on. If `None`, a new figure
            and axes are created.

        cmap : str, optional
            The matplotlib colormap to use. The default is ``'viridis'``.

        Returns
        -------
        fig, ax : tuple
            The matplotlib figure and axes containing the plot.
        """
        import matplotlib.pyplot as plt

        trust_map = getattr(self, 'trust_map', None)
        if trust_map is None:
            raise ValueError('No trust map available in this model')

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = ax.figure

        im = ax.imshow(trust_map, origin='lower', cmap=cmap, aspect='equal')
        ax.set_xlabel('Oversampled X grid')
        ax.set_ylabel('Oversampled Y grid')
        ax.set_title('Trust Map (RMS Residual per Grid Cell)')
        fig.colorbar(im, ax=ax, label='RMS Residual')
        return fig, ax

    def plot_model_with_trust_map(self, x, y, *, grid_factor=1, ax=None, cmap='viridis'):
        """
        Plot the local ePSF and its trust map side-by-side at detector position (x, y).

        Parameters
        ----------
        x, y : float
            Detector position where the local ePSF should be evaluated.

        grid_factor : int, optional
            The factor by which to refine the evaluation grid relative to
            the native oversampled ePSF grid. The default is 1.

        ax : `~matplotlib.axes.Axes`, optional
            Ignored; a new figure with 2 subplots is always created.

        cmap : str, optional
            The matplotlib colormap to use. The default is ``'viridis'``.

        Returns
        -------
        fig, axes : tuple
            The matplotlib figure and axes array (one for ePSF, one for trust map).
        """
        import matplotlib.pyplot as plt

        trust_map = getattr(self, 'trust_map', None)
        if trust_map is None:
            raise ValueError('No trust map available in this model')

        local_epsf = self.make_image_psf(x, y)
        half_x = (self.shape[1] - 1) / (2.0 * self.oversampling[1])
        half_y = (self.shape[0] - 1) / (2.0 * self.oversampling[0])

        grid_factor = int(grid_factor)
        if grid_factor <= 0:
            raise ValueError('grid_factor must be a positive integer')

        nx = self.shape[1] * grid_factor
        ny = self.shape[0] * grid_factor
        xgrid = np.linspace(-half_x, half_x, nx)
        ygrid = np.linspace(-half_y, half_y, ny)
        yy, xx = np.meshgrid(ygrid, xgrid, indexing='ij')
        values = local_epsf.evaluate(x=xx, y=yy, flux=1.0, x_0=0.0, y_0=0.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                 constrained_layout=True)

        im_epsf = axes[0].imshow(values, origin='lower',
                                 extent=(-half_x, half_x, -half_y, half_y),
                                 cmap=cmap, aspect='equal')
        axes[0].set_xlabel('Model x (pixels)')
        axes[0].set_ylabel('Model y (pixels)')
        axes[0].set_title(f'Local ePSF at detector position ({x:.2f}, {y:.2f})')
        fig.colorbar(im_epsf, ax=axes[0], label='ePSF value')

        im_trust = axes[1].imshow(trust_map, origin='lower', cmap=cmap, aspect='equal')
        axes[1].set_xlabel('Oversampled X grid')
        axes[1].set_ylabel('Oversampled Y grid')
        axes[1].set_title('Trust Map (RMS Residual)')
        fig.colorbar(im_trust, ax=axes[1], label='RMS Residual')

        return fig, axes

    def _local_epsf_normalization(self, data):
        """
        Return the native-pixel-center normalization factor for a local
        oversampled ePSF image.
        """
        epsf = self.epsf_class(data=data, oversampling=self.oversampling,
                               origin=self.origin, fill_value=self.fill_value)

        box_size = np.asarray(np.asarray(data.shape) / self.oversampling,
                              dtype=int)
        box_size = np.where(box_size < 3, 3, box_size)
        half_x = int(np.floor(box_size[1] / 2))
        half_y = int(np.floor(box_size[0] / 2))
        norm_x = np.arange(-half_x, half_x + 1)
        norm_y = np.arange(-half_y, half_y + 1)
        yy, xx = np.meshgrid(norm_y, norm_x)
        vals = epsf.evaluate(x=xx, y=yy, flux=1.0, x_0=0.0, y_0=0.0)
        return np.nansum(vals)

    def _normalise_local_epsf_data(self, data):
        """
        Normalize a local oversampled ePSF image so that its values
        sampled at native pixel centers sum to unity.
        """
        total = self._local_epsf_normalization(data)
        if np.isfinite(total) and total > 0.0:
            return data / total

        return data

    def deepcopy(self):
        return copy.deepcopy(self)


class SpatialPPEMapModel:
    """
    Detector-position-dependent PPE correction model.

    Each phase-bin value in the flux/x/y PPE maps is represented as a
    low-order polynomial in detector position.
    """

    def __init__(self, flux_coeff, x_coeff, y_coeff, *, supersampling,
                 detector_shape, degree=1, detector_origin=None,
                 detector_span=None):
        self.flux_coeff = np.asanyarray(flux_coeff, dtype=float)
        self.x_coeff = np.asanyarray(x_coeff, dtype=float)
        self.y_coeff = np.asanyarray(y_coeff, dtype=float)
        self.supersampling = as_pair('supersampling', supersampling,
                                     lower_bound=(1, 1))
        self.detector_shape = as_pair('detector_shape', detector_shape,
                                      lower_bound=(2, 2))
        if detector_origin is None:
            detector_origin = (0.0, 0.0)
        self.detector_origin = np.asanyarray(detector_origin, dtype=float)
        if self.detector_origin.shape != (2,):
            raise ValueError('detector_origin must have shape (2,)')
        if not np.all(np.isfinite(self.detector_origin)):
            raise ValueError('detector_origin must contain only finite values')
        if detector_span is None:
            detector_span = self.detector_shape - 1
        self.detector_span = np.asanyarray(detector_span, dtype=float)
        if self.detector_span.shape != (2,):
            raise ValueError('detector_span must have shape (2,)')
        if not np.all(np.isfinite(self.detector_span)):
            raise ValueError('detector_span must contain only finite values')
        self.degree = int(degree)
        if self.degree not in (0, 1, 2):
            raise ValueError('degree must be 0, 1, or 2')

        expected_basis = len(SpatialEPSFModel._basis_labels_for_degree(
            self.degree))
        for coeff in (self.flux_coeff, self.x_coeff, self.y_coeff):
            if coeff.ndim != 3:
                raise ValueError('PPE coefficient arrays must be 3D')
            if coeff.shape[0] != expected_basis:
                raise ValueError('PPE coefficient array first axis does not '
                                 'match degree')
            if tuple(coeff.shape[1:]) != tuple(self.supersampling):
                raise ValueError('PPE coefficient grid shape does not match '
                                 'supersampling')

    def basis_vector(self, x, y):
        dummy = SpatialEPSFModel(np.zeros((len(
            SpatialEPSFModel._basis_labels_for_degree(self.degree)), 3, 3)),
            oversampling=(1, 1), detector_shape=self.detector_shape,
            degree=self.degree, detector_origin=self.detector_origin,
            detector_span=self.detector_span)
        return dummy.basis_vector(x, y)

    def local_maps(self, x, y):
        basis = self.basis_vector(x, y)
        flux_map = np.tensordot(basis, self.flux_coeff, axes=(0, 0))
        x_map = np.tensordot(basis, self.x_coeff, axes=(0, 0))
        y_map = np.tensordot(basis, self.y_coeff, axes=(0, 0))
        return PPEMap(self.supersampling, flux_map, x_map, y_map)

    def __call__(self, flux, x, y, *, det_x=None, det_y=None):
        if det_x is None:
            det_x = x
        if det_y is None:
            det_y = y
        local_map = self.local_maps(det_x, det_y)
        return local_map(flux, x, y)


class SpatialEPSFFitter:
    """
    Fit a spatially varying ePSF model to stars.

    This mirrors the API of the classic `EPSFFitter`, but generates a
    local `ImagePSF` at each star's detector position before fitting.
    """

    def __init__(self, *, fitter=None, fit_boxsize=3, progress_bar=False,
                 plot_fit_checks=False, model_weight_map=None,
                 model_weight_maxiters=1, model_weight_center_tol=1.0e-3,
                 **fitter_kwargs):
        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        self.fit_boxsize = None if fit_boxsize is None else tuple(
            int(val) for val in as_pair('fit_boxsize', fit_boxsize,
                                        lower_bound=(3, 0), check_odd=True))
        self.progress_bar = bool(progress_bar)
        self.plot_fit_checks = bool(plot_fit_checks)

        # Temporary test of the effect of model_weight_map:
        if model_weight_map is None:
            alpha = 0.0

            def _default_model_weight_map(local_epsf, star):
                trust_map = getattr(local_epsf, 'trust_map', None)
                if trust_map is None:
                    return np.ones_like(local_epsf.data, dtype=float)

                trust_map = np.asanyarray(trust_map, dtype=float)
                valid = np.isfinite(trust_map) & (trust_map > 0.0)

                weight_map = np.ones_like(trust_map, dtype=float)
                weight_map[valid] = 1.0 / np.power(trust_map[valid], alpha)

                median_weight = np.nanmedian(weight_map[valid])
                if np.isfinite(median_weight) and median_weight > 0.0:
                    weight_map[valid] /= median_weight

                return weight_map

            self.model_weight_map = _default_model_weight_map
        else:
            self.model_weight_map = model_weight_map

        self.model_weight_maxiters = int(model_weight_maxiters)
        if self.model_weight_maxiters <= 0:
            raise ValueError('model_weight_maxiters must be a positive '
                             'integer')

        self.model_weight_center_tol = float(model_weight_center_tol)
        if self.model_weight_center_tol < 0.0:
            raise ValueError('model_weight_center_tol must be >= 0')

        remove_kwargs = ['x', 'y', 'z', 'weights']
        self.fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            self.fitter_kwargs.pop(kwarg, None)

    @staticmethod
    def _get_star_fit_data(star, fit_boxsize):
        if fit_boxsize is not None:
            xcenter, ycenter = star.cutout_center
            large_slc, _ = overlap_slices(star.shape, fit_boxsize,
                                          (ycenter, xcenter),
                                          mode='strict')

            data = star.data[large_slc]
            weights = star.weights[large_slc]
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            data = star.data
            weights = star.weights
            x0 = 0
            y0 = 0

        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]
        return data, weights, xx, yy

    def _resolve_model_weight_map(self, local_epsf, star):
        if self.model_weight_map is None:
            trust_map = getattr(local_epsf, 'trust_map', None)
            if trust_map is None:
                return None

            trust_map = np.asanyarray(trust_map, dtype=float)
            if trust_map.shape != local_epsf.data.shape:
                raise ValueError('local trust_map must have the same shape '
                                 'as the local ePSF model data')

            valid = np.isfinite(trust_map) & (trust_map > 0.0)
            if not np.any(valid):
                return np.zeros_like(trust_map, dtype=float)

            weight_map = np.zeros_like(trust_map, dtype=float)
            weight_map[valid] = 1.0 / trust_map[valid]
            median_weight = np.nanmedian(weight_map[valid])
            if np.isfinite(median_weight) and median_weight > 0.0:
                weight_map[valid] /= median_weight
            return weight_map

        if callable(self.model_weight_map):
            weight_map = self.model_weight_map(local_epsf, star)
        else:
            weight_map = self.model_weight_map

        if weight_map is None:
            return None

        weight_map = np.asanyarray(weight_map, dtype=float)
        if weight_map.shape != local_epsf.data.shape:
            raise ValueError('model_weight_map must have the same shape as '
                             'the local ePSF model data')

        return weight_map

    def _evaluate_model_weights(self, local_epsf, weight_map, xx, yy):
        model_weights = ImagePSF(data=weight_map,
                                 oversampling=local_epsf.oversampling,
                                 origin=local_epsf.origin,
                                 fill_value=0.0)
        values = model_weights.evaluate(x=xx, y=yy, flux=1.0,
                                        x_0=0.0, y_0=0.0)
        values = np.asanyarray(values, dtype=float)
        values[~np.isfinite(values)] = 0.0
        return np.clip(values, 0.0, None)

    def __call__(self, spatial_epsf, stars):
        if len(stars) == 0:
            return stars
        if not isinstance(spatial_epsf, SpatialEPSFModel):
            raise TypeError('spatial_epsf must be a SpatialEPSFModel')

        fitted_stars = []
        pbar = None
        if self.progress_bar:
            pbar = add_progress_bar(total=len(stars),
                                    desc='SpatialEPSFFitter')

        for item in stars:
            if isinstance(item, EPSFStar):
                fitted_stars.append(self._fit_star(spatial_epsf, item))
            elif isinstance(item, LinkedEPSFStar):
                linked = [self._fit_star(spatial_epsf, star, make_plot=False)
                          for star in item]
                if self.plot_fit_checks:
                    for fitted_star in linked:
                        if getattr(fitted_star, '_fit_error_status', 0) != 2:
                            local_epsf = spatial_epsf.make_image_psf(
                                fitted_star.center[0], fitted_star.center[1])
                            self._plot_fit_check(local_epsf, fitted_star)
                            break
                fitted_stars.append(LinkedEPSFStar(linked))
            else:
                if pbar is not None:
                    pbar.close()
                raise TypeError('stars must contain only EPSFStar and/or '
                                'LinkedEPSFStar objects')

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

        return EPSFStars(fitted_stars)

    def _fit_star(self, spatial_epsf, star, *, make_plot=True):
        star_work = copy.deepcopy(star)
        fit_error_status = 0
        fit_info = None
        last_local_epsf = None

        maxiters = 1
        if (self.model_weight_map is not None
            or getattr(spatial_epsf, 'trust_map', None) is not None):
            maxiters = self.model_weight_maxiters

        for _ in range(maxiters):
            local_epsf = spatial_epsf.make_image_psf(star_work.center[0],
                                                     star_work.center[1])
            last_local_epsf = local_epsf

            try:
                data, weights, xx, yy = self._get_star_fit_data(
                    star_work, self.fit_boxsize)
            except (PartialOverlapError, NoOverlapError):
                warnings.warn(f'The star at ({star.center[0]}, '
                              f'{star.center[1]}) cannot be fit because '
                              'its fitting region extends beyond the star '
                              'cutout image.', AstropyUserWarning)
                star_bad = copy.deepcopy(star)
                star_bad._fit_error_status = 1
                return star_bad

            fit_weights = weights
            model_weight_map = self._resolve_model_weight_map(local_epsf,
                                                              star_work)
            if model_weight_map is not None:
                model_weights = self._evaluate_model_weights(
                    local_epsf, model_weight_map, xx, yy)
                fit_weights = np.asanyarray(weights, dtype=float) * model_weights

            local_epsf.flux = star_work.flux
            local_epsf.x_0 = 0.0
            local_epsf.y_0 = 0.0

            try:
                fitted_epsf = self.fitter(model=local_epsf, x=xx, y=yy,
                                          z=data, weights=fit_weights,
                                          **self.fitter_kwargs)
            except TypeError:
                fitted_epsf = self.fitter(model=local_epsf, x=xx, y=yy,
                                          z=data, **self.fitter_kwargs)

            fit_error_status = 0
            if self.fitter_has_fit_info:
                fit_info = copy.copy(self.fitter.fit_info)
                if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                    fit_error_status = 2
            else:
                fit_info = None

            if fit_error_status == 2:
                break

            x_shift = fitted_epsf.x_0.value
            y_shift = fitted_epsf.y_0.value
            x_center = star_work.cutout_center[0] + x_shift
            y_center = star_work.cutout_center[1] + y_shift
            star_work.cutout_center = (x_center, y_center)
            star_work.flux = fitted_epsf.flux.value
            star_work._fit_info = fit_info

            if self.model_weight_map is None:
                break

            if np.hypot(x_shift, y_shift) <= self.model_weight_center_tol:
                break

        if fit_error_status != 2:
            star_work._fit_error_status = 0
            fitted_star = star_work
        else:
            fitted_star = copy.deepcopy(star)
            fitted_star._fit_error_status = fit_error_status

        if self.plot_fit_checks and make_plot and fit_error_status != 2:
            self._plot_fit_check(last_local_epsf, fitted_star)

        return fitted_star

    def _plot_fit_check(self, local_epsf, star):
        """
        Plot the normalized local ePSF, the star data, and the fitted
        registered model for temporary inspection, together with the
        residual image.
        """
        import matplotlib.pyplot as plt

        model_img = star.register_epsf(local_epsf)
        residual_img = star.data - model_img
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                                 constrained_layout=True)
        panels = (
            (local_epsf.data, 'Normalized Local ePSF'),
            (star.data, 'Star Data'),
            (model_img, 'Fitted Registered Model'),
            (residual_img, 'Residual (Data - Model)'),
        )

        for ax, (data, title) in zip(axes, panels):
            cmap = 'coolwarm' if 'Residual' in title else 'viridis'
            im = ax.imshow(data, origin='lower', cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel('X Pixel')
            ax.set_ylabel('Y Pixel')
            fig.colorbar(im, ax=ax)

        plt.show()


class SpatialEPSFBuilder:
    """
    Build a spatially varying ePSF where each oversampled grid point is
    modeled as a low-order polynomial in detector position.

    Linked stars can optionally be constrained after each fit iteration
    to share a common mean sky position and/or mean flux before the
    residual stack is formed.

    If ``detector_shape`` is not provided, then the detector-coordinate
    normalization is inferred from the distribution of the input star
    centers and used as an approximation to the sampled detector region.
    """

    def __init__(self, *,
                 oversampling=4, 
                 shape=None,
                 degree=1, 
                 epsf_class=ImagePSF,
                 fitter=None, 
                 maxiters=10, 
                 progress_bar=False,
                 center_accuracy=1.0e-3,
                 detector_shape=None,
                 smoothing_kernel='quartic',
                 residual_smoothing_kernel='gaussian',
                 recenter_epsf=True,
                 normalise_epsf=True,
                 calibrate_ppe=('Flux', 'Position'),
                 constrain_stars=('Flux', 'Position'),
                 residual_star_rms_clip=None,
                 residual_outlier_clip=3.0,
                 residual_min_valid_samples=10,
                 interpolate_missing_coefficients=True,
                 coefficient_interpolation_method='cubic',
                 plot_diagnostics=False,
                 ):
        self._detector_shape_explicit = detector_shape is not None
        self.detector_shape = (None if detector_shape is None else
                               as_pair('detector_shape', detector_shape,
                                       lower_bound=(2, 2)))
        self.detector_origin = None
        self.detector_span = None
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        self.shape = None if shape is None else as_pair('shape', shape,
                                                        lower_bound=(0, 1))
        self.degree = int(degree)
        if self.degree not in (0, 1, 2):
            raise ValueError('degree must be 0, 1, or 2')

        self.fitter = fitter if fitter is not None else SpatialEPSFFitter()
        if not isinstance(self.fitter, SpatialEPSFFitter):
            raise TypeError('fitter must be a SpatialEPSFFitter instance')
        self.epsf_class = epsf_class
        if isinstance(self.epsf_class, partial):
            candidate = self.epsf_class.func
        else:
            candidate = self.epsf_class
        if not issubclass(candidate, ImagePSF):
            raise TypeError('epsf_class must be a subclass of ImagePSF')

        self.maxiters = int(maxiters)
        if self.maxiters <= 0:
            raise ValueError('maxiters must be a positive number')

        self.progress_bar = bool(progress_bar)
        self.fitter.progress_bar = self.progress_bar
        self.recenter_epsf = bool(recenter_epsf)
        self.normalise_epsf = bool(normalise_epsf)
        self.center_accuracy_sq = float(center_accuracy)**2
        self.residual_update_fraction = 0.8

        self.residual_min_valid_samples = int(residual_min_valid_samples)
        if self.residual_min_valid_samples <= 0:
            raise ValueError('residual_min_valid_samples must be positive')

        self.interpolate_missing_coefficients = bool(
            interpolate_missing_coefficients)
        self.coefficient_interpolation_method = coefficient_interpolation_method
        if self.coefficient_interpolation_method not in ('cubic', 'nearest'):
            raise ValueError("coefficient_interpolation_method must be "
                             "'cubic' or 'nearest'")

        if isinstance(calibrate_ppe, str):
            calibrate_ppe = (calibrate_ppe,)
        try:
            calibrate_ppe = tuple(calibrate_ppe)
        except TypeError as exc:
            raise TypeError('calibrate_ppe must be an iterable containing '
                            "'Flux' and/or 'Position'") from exc

        allowed_ppe = {'Flux', 'Position'}
        invalid_ppe = [item for item in calibrate_ppe if item not in allowed_ppe]
        if invalid_ppe:
            raise ValueError("calibrate_ppe entries must be 'Flux' and/or "
                             "'Position'")
        self.calibrate_ppe = tuple(dict.fromkeys(calibrate_ppe))

        if isinstance(constrain_stars, str):
            constrain_stars = (constrain_stars,)
        try:
            constrain_stars = tuple(constrain_stars)
        except TypeError as exc:
            raise TypeError('constrain_stars must be an iterable containing '
                            "'Flux' and/or 'Position'") from exc

        allowed_constraints = {'Flux', 'Position'}
        invalid_constraints = [item for item in constrain_stars
                               if item not in allowed_constraints]
        if invalid_constraints:
            raise ValueError("constrain_stars entries must be 'Flux' and/or "
                             "'Position'")
        self.constrain_stars = tuple(dict.fromkeys(constrain_stars))

        self.plot_diagnostics = bool(plot_diagnostics)
        self.trust_map_floor = 1.0e-6

        if (residual_star_rms_clip is not None
                and residual_star_rms_clip <= 0):
            raise ValueError('residual_star_rms_clip must be positive or None')
        self.residual_star_rms_clip = residual_star_rms_clip

        self.ppe_degree = self.degree
        self.ppe_supersampling = self.oversampling
        self.ppe_min_valid_samples = self.residual_min_valid_samples

        if (residual_outlier_clip is not None
                and residual_outlier_clip <= 0):
            raise ValueError('residual_outlier_clip must be positive or None')
        self.residual_outlier_clip = residual_outlier_clip
        self._sigma_clip = None if residual_outlier_clip is None else SigmaClip(
            sigma=float(residual_outlier_clip), maxiters=10)

        self.smoothing_kernel = smoothing_kernel
        self.residual_smoothing_kernel = residual_smoothing_kernel
        self._smooth_kernel = self._make_smoothing_kernel(smoothing_kernel)
        self._residual_smooth_kernel = self._make_residual_smoothing_kernel(
            residual_smoothing_kernel)

        self._models = []
        self._ppe_models = []
        self.final_ppe_model = None

    def _set_detector_geometry(self, detector_shape, detector_origin,
                               detector_span):
        self.detector_shape = as_pair('detector_shape', detector_shape,
                                      lower_bound=(2, 2))
        self.detector_origin = np.asanyarray(detector_origin, dtype=float)
        self.detector_span = np.asanyarray(detector_span, dtype=float)

    def _resolve_detector_geometry(self, stars, *, init_model=None):
        """
        Resolve the detector-coordinate normalization used by the
        spatial polynomial basis.
        """
        if self._detector_shape_explicit:
            if self.detector_origin is None:
                self.detector_origin = np.zeros(2, dtype=float)
            if self.detector_span is None:
                self.detector_span = self.detector_shape - 1
            return

        if init_model is not None:
            detector_origin = getattr(init_model, 'detector_origin',
                                      np.zeros(2, dtype=float))
            detector_span = getattr(init_model, 'detector_span',
                                    init_model.detector_shape - 1)
            self._set_detector_geometry(init_model.detector_shape,
                                        detector_origin, detector_span)
            return

        det_results = self._collect_detector_position_results(stars)
        det_x = np.asarray(det_results['det_x'], dtype=float)
        det_y = np.asarray(det_results['det_y'], dtype=float)
        valid = np.isfinite(det_x) & np.isfinite(det_y)
        if np.count_nonzero(valid) == 0:
            raise ValueError('No valid detector-position samples were found')

        xmin = np.nanmin(det_x[valid])
        xmax = np.nanmax(det_x[valid])
        ymin = np.nanmin(det_y[valid])
        ymax = np.nanmax(det_y[valid])

        xspan = xmax - xmin
        yspan = ymax - ymin
        xorigin = xmin
        yorigin = ymin
        if xspan <= 0.0:
            xspan = 1.0
            xorigin = xmin - 0.5 * xspan
        if yspan <= 0.0:
            yspan = 1.0
            yorigin = ymin - 0.5 * yspan

        detector_shape = (int(np.ceil(yspan)) + 1, int(np.ceil(xspan)) + 1)
        self._set_detector_geometry(detector_shape, (yorigin, xorigin),
                                    (yspan, xspan))

    def _log(self, message):
        """
        Emit a simple terminal progress message when enabled.
        """
        if self.progress_bar:
            print(message, flush=True)

    def _apply_linked_constraints(self, stars):
        """
        Constrain linked-star fluxes and/or centers using the mean values
        of the members within each linked group.
        """
        constrained = copy.deepcopy(stars)
        if 'Position' in self.constrain_stars:
            constrained.constrain_linked_centres()
        if 'Flux' in self.constrain_stars:
            constrained.constrain_linked_fluxes()
        return constrained

    def _make_empty_ppe_model(self):
        nbasis = len(SpatialEPSFModel._basis_labels_for_degree(
            self.ppe_degree))
        shape = (nbasis, self.ppe_supersampling[0], self.ppe_supersampling[1])
        zeros = np.zeros(shape, dtype=float)
        return SpatialPPEMapModel(zeros, zeros.copy(), zeros.copy(),
                                  supersampling=self.ppe_supersampling,
                                  detector_shape=self.detector_shape,
                                  degree=self.ppe_degree,
                                  detector_origin=self.detector_origin,
                                  detector_span=self.detector_span)

    @staticmethod
    def _make_smoothing_kernel(kernel):
        if kernel is None:
            return None
        if isinstance(kernel, np.ndarray):
            return kernel
        if kernel == 'quartic':
            return np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])
        if kernel == 'quadratic':
            return np.array(
                [[-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [+0.03999952, 0.12571449, 0.15428215, 0.12571449,
                  +0.03999952],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311]])
        raise ValueError("Unsupported smoothing_kernel")

    def _make_residual_smoothing_kernel(self, kernel):
        if kernel is None:
            return None
        if isinstance(kernel, np.ndarray):
            return kernel
        if kernel == 'gaussian':
            return Gaussian2DKernel(x_stddev=1, y_stddev=1,
                                    x_size=self.oversampling[1],
                                    y_size=self.oversampling[0])
        raise ValueError("Unsupported residual_smoothing_kernel")

    def __call__(self, stars):
        return self.build_epsf(stars)

    def _n_basis(self):
        return len(SpatialEPSFModel._basis_labels_for_degree(self.degree))

    def _create_initial_model(self, stars):
        if self.shape is not None:
            shape = as_pair('shape', self.shape, lower_bound=(0, 1),
                            check_odd=True)
        else:
            x_shape = (np.ceil(stars._max_shape[1]) * self.oversampling[1]
                       + 1).astype(int)
            y_shape = (np.ceil(stars._max_shape[0]) * self.oversampling[0]
                       + 1).astype(int)
            shape = np.array((y_shape, x_shape))

        shape = [(i + 1) if i % 2 == 0 else i for i in shape]
        coeff_data = np.zeros((self._n_basis(), shape[0], shape[1]),
                              dtype=float)

        # Start from a simple average star profile in the constant term.
        stack = []
        for star in stars.all_good_stars:
            data = np.array(star.data, copy=True, dtype=float)
            data = data / max(star.flux, 1.0e-12)
            stack.append(data)

        if len(stack) > 0:
            ref = nanmedian(np.array(stack), axis=0)
            ref_oversampled = self._oversample_reference(ref, tuple(shape))
            coeff_data[0] = ref_oversampled

        return SpatialEPSFModel(coeff_data, oversampling=self.oversampling,
                                detector_shape=self.detector_shape,
                                degree=self.degree,
                                detector_origin=self.detector_origin,
                                detector_span=self.detector_span,
                                normalize_local_epsf=self.normalise_epsf,
                                trust_map=None,
                                epsf_class=self.epsf_class)

    def _compute_trust_map_from_residuals(self, residuals, weights):
        """
        Compute an RMS residual map on the oversampled ePSF grid.

        The returned map has one RMS value per oversampled grid section,
        using all finite residual samples that map into that section.
        """
        residuals = np.asanyarray(residuals, dtype=float)
        weights = np.asanyarray(weights, dtype=float)

        valid = (np.isfinite(residuals) & np.isfinite(weights)
                 & (weights > 0.0))
        if residuals.ndim != 3:
            raise ValueError('residuals must be a 3D array')
        if weights.shape != residuals.shape:
            raise ValueError('weights must have the same shape as residuals')

        sumsqs = np.sum(np.where(valid, residuals**2, 0.0), axis=0)
        counts = np.sum(valid, axis=0)
        trust_map = np.full(residuals.shape[1:], np.nan, dtype=float)
        good = counts > 0
        trust_map[good] = np.sqrt(sumsqs[good] / counts[good])

        fill = np.nanmedian(trust_map[good]) if np.any(good) else 1.0
        if not np.isfinite(fill) or fill <= 0.0:
            fill = 1.0
        trust_map[~np.isfinite(trust_map)] = fill
        trust_map = np.clip(trust_map, self.trust_map_floor, None)
        return trust_map

    def _oversample_reference(self, ref_data, target_shape):
        """
        Interpolate a native-grid reference image onto the oversampled
        ePSF grid used by the spatial model.
        """
        ref_data = np.asanyarray(ref_data, dtype=float)
        target_shape = np.asarray(target_shape, dtype=int)

        if np.any(target_shape <= 0):
            raise ValueError('target_shape must be positive')

        y_in = np.arange(ref_data.shape[0], dtype=float)
        x_in = np.arange(ref_data.shape[1], dtype=float)
        y_out = np.linspace(0.0, ref_data.shape[0] - 1.0, target_shape[0])
        x_out = np.linspace(0.0, ref_data.shape[1] - 1.0, target_shape[1])
        yy_out, xx_out = np.meshgrid(y_out, x_out, indexing='ij')

        coords = np.vstack((yy_out.ravel(), xx_out.ravel()))
        oversampled = map_coordinates(ref_data, coords, order=3,
                                      mode='nearest')
        oversampled = oversampled.reshape(tuple(target_shape))

        oversampled[~np.isfinite(oversampled)] = 0.0
        return oversampled

    def _resample_residual(self, star, spatial_model):
        local_epsf = spatial_model.make_image_psf(star.center[0], star.center[1])
        residual_img = star.compute_residual_image(local_epsf)
        residual_img /= max(star.flux, 1.0e-12)
        residual_img = residual_img[~star.mask].ravel()

        x = spatial_model.oversampling[1] * star._xidx_centered
        y = spatial_model.oversampling[0] * star._yidx_centered

        epsf_xcenter = int((spatial_model.shape[1] - 1) / 2)
        epsf_ycenter = int((spatial_model.shape[0] - 1) / 2)

        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)
        x_coord = x + epsf_xcenter - xidx
        y_coord = y + epsf_ycenter - yidx
        xdist = np.abs(x_coord) / 0.5
        ydist = np.abs(y_coord) / 0.5

        resampled_img = np.full(spatial_model.shape, np.nan)
        img_weights = np.full(spatial_model.shape, 0.0)
        x_coords_img = np.full(spatial_model.shape, np.nan)
        y_coords_img = np.full(spatial_model.shape, np.nan)

        mask = ((xidx >= 0) & (xidx < spatial_model.shape[1])
                & (yidx >= 0) & (yidx < spatial_model.shape[0]))
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]
        xdist_ = xdist[mask]
        ydist_ = ydist[mask]
        x_coord_ = x_coord[mask]
        y_coord_ = y_coord[mask]

        resampled_img[yidx_, xidx_] = residual_img[mask]
        img_weights[yidx_, xidx_] = (1.0
                                     - (np.sqrt(xdist_**2 + ydist_**2)
                                        / np.sqrt(2.0)))
        x_coords_img[yidx_, xidx_] = x_coord_
        y_coords_img[yidx_, xidx_] = y_coord_

        return resampled_img, img_weights, x_coords_img, y_coords_img

    def _resample_residuals(self, stars, spatial_model):
        nstars = stars.n_good_stars
        shape = (nstars, spatial_model.shape[0], spatial_model.shape[1])
        residuals = np.zeros(shape)
        weights = np.zeros(shape)
        x_coords = np.zeros(shape)
        y_coords = np.zeros(shape)
        det_x = np.zeros(nstars)
        det_y = np.zeros(nstars)
        group_id = np.full(nstars, -1, dtype=int)

        i = 0
        linked_id = 0
        for item in stars:
            if isinstance(item, LinkedEPSFStar):
                for star in item.all_good_stars:
                    resampled = self._resample_residual(star, spatial_model)
                    residuals[i], weights[i], x_coords[i], y_coords[i] = resampled
                    det_x[i] = star.center[0]
                    det_y[i] = star.center[1]
                    group_id[i] = linked_id
                    i += 1
                linked_id += 1
                continue

            if item._excluded_from_fit:
                continue
            resampled = self._resample_residual(item, spatial_model)
            residuals[i], weights[i], x_coords[i], y_coords[i] = resampled
            det_x[i] = item.center[0]
            det_y[i] = item.center[1]
            i += 1

        return residuals, weights, x_coords, y_coords, det_x, det_y, group_id

    @staticmethod
    def _mad_std(data):
        data = np.asanyarray(data, dtype=float)
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        return 1.482602218505602 * mad

    def _select_residual_stars(self, stars, spatial_model):
        """
        Exclude whole stars from the residual stack using a robust RMS
        clip against the current spatial ePSF model.
        """
        if self.residual_star_rms_clip is None:
            return copy.deepcopy(stars)

        selected = copy.deepcopy(stars)
        good_stars = selected.all_good_stars
        if len(good_stars) == 0:
            return selected

        rms_values = []
        for star in good_stars:
            local_epsf = spatial_model.make_image_psf(star.center[0],
                                                      star.center[1])
            residual = star.compute_residual_image(local_epsf)
            residual = residual / max(star.flux, 1.0e-12)
            residual = residual[~star.mask]
            residual = residual[np.isfinite(residual)]
            if residual.size == 0:
                rms_values.append(np.nan)
            else:
                rms_values.append(np.sqrt(np.nanmean(residual**2)))

        rms_values = np.asarray(rms_values, dtype=float)
        valid = np.isfinite(rms_values)
        if np.count_nonzero(valid) < 3:
            return selected

        center = np.nanmedian(rms_values[valid])
        scale = self._mad_std(rms_values[valid])
        if not np.isfinite(scale) or scale == 0.0:
            return selected

        threshold = center + self.residual_star_rms_clip * scale
        for star, rms in zip(good_stars, rms_values, strict=True):
            if not np.isfinite(rms) or rms > threshold:
                star._excluded_from_fit = True

        if selected.n_good_stars == 0:
            return copy.deepcopy(stars)
        return selected

    @staticmethod
    def _auto_quadratic_offset_core_size(shape):
        """
        Infer the quadratic-offset core size from the oversampled ePSF
        grid shape using roughly one-third of the footprint in each
        dimension, forced to odd values with a minimum of 3.
        """
        shape = np.asanyarray(shape, dtype=int)
        if shape.shape != (2,):
            raise ValueError('shape must have shape (2,)')

        core = np.rint(shape / 3.0).astype(int)
        core = np.maximum(core, 3)
        core = np.where(core % 2 == 0, core + 1, core)
        max_core = np.where(shape % 2 == 0, shape - 1, shape)
        max_core = np.maximum(max_core, 3)
        return np.minimum(core, max_core)

    def _interpolate_missing_coefficient_images(self, coeff_data):
        if not self.interpolate_missing_coefficients:
            coeff_data = np.array(coeff_data, copy=True, dtype=float)
            coeff_data[~np.isfinite(coeff_data)] = 0.0
            return coeff_data

        coeff_data = np.array(coeff_data, copy=True, dtype=float)
        for idx in range(coeff_data.shape[0]):
            image = coeff_data[idx]
            mask = ~np.isfinite(image)
            if not np.any(mask):
                continue

            if np.all(mask):
                image[:] = 0.0
                continue

            method = self.coefficient_interpolation_method
            if method == 'cubic' and np.count_nonzero(~mask) < 3:
                method = 'nearest'

            try:
                image = _interpolate_missing_data(image, mask=mask,
                                                  method=method)
            except (ValueError, RuntimeError, QhullError):
                image = _interpolate_missing_data(coeff_data[idx], mask=mask,
                                                  method='nearest')

            remaining = ~np.isfinite(image)
            if np.any(remaining):
                image = _interpolate_missing_data(image, mask=remaining,
                                                  method='nearest')
            coeff_data[idx] = image

        coeff_data[~np.isfinite(coeff_data)] = 0.0
        return coeff_data

    def _underconstrained_residual_mask(self, residuals, x_coords=None,
                                        y_coords=None):
        _, ny, nx = residuals.shape
        mask = np.zeros((ny, nx), dtype=bool)
        use_offsets = x_coords is not None and y_coords is not None
        for iy in range(ny):
            for ix in range(nx):
                valid = np.isfinite(residuals[:, iy, ix])
                if use_offsets:
                    valid &= (np.isfinite(x_coords[:, iy, ix])
                              & np.isfinite(y_coords[:, iy, ix]))
                mask[iy, ix] = (np.count_nonzero(valid)
                                < self.residual_min_valid_samples)
        return mask

    def _fit_residual_coefficients(self, residuals, det_x, det_y,
                                   x_coords=None, y_coords=None):
        """
        Fit detector-position-dependent residual coefficient surfaces.

        If ``x_coords`` and ``y_coords`` are provided, then each
        grid-point fit also includes within-cell offset terms. By
        default the fit uses linear offset terms everywhere, and a
        quadratic offset model within the central core defined by
        ``quadratic_offset_core_size``.
        """
        nbasis = self._n_basis()
        basis = SpatialEPSFModel(np.zeros((nbasis, 3, 3)),
                                 oversampling=self.oversampling,
                                 detector_shape=self.detector_shape,
                                 degree=self.degree,
                                 detector_origin=self.detector_origin,
                                 detector_span=self.detector_span,
                                 epsf_class=self.epsf_class).basis_vector(
                                     det_x, det_y)
        _, ny, nx = residuals.shape
        coeff_update = np.full((nbasis, ny, nx), np.nan, dtype=float)
        use_offsets = x_coords is not None and y_coords is not None
        xcenter = nx // 2
        ycenter = ny // 2
        quadratic_offset_core_size = self._auto_quadratic_offset_core_size(
            (ny, nx))
        half_core_x = quadratic_offset_core_size[1] // 2
        half_core_y = quadratic_offset_core_size[0] // 2

        for iy in range(ny):
            for ix in range(nx):
                z = residuals[:, iy, ix]
                valid = np.isfinite(z)
                if use_offsets:
                    dx = x_coords[:, iy, ix]
                    dy = y_coords[:, iy, ix]
                    valid &= np.isfinite(dx) & np.isfinite(dy)
                if np.count_nonzero(valid) < self.residual_min_valid_samples:
                    continue

                z_valid = z[valid]
                design = basis[:, valid].T
                if use_offsets:
                    dx_valid = dx[valid]
                    dy_valid = dy[valid]
                    offset_terms = self._residual_offset_terms(
                        dx_valid, dy_valid, ix, iy, nx, ny)
                    design = np.column_stack((design, *offset_terms))

                if self._sigma_clip is not None:
                    clipped = self._sigma_clip(z_valid, axis=0, masked=True,
                                               return_bounds=False)
                    keep = ~clipped.mask
                    z_fit = clipped.data[keep]
                    design = design[keep]
                else:
                    z_fit = z_valid

                if z_fit.size < self.residual_min_valid_samples:
                    continue

                coeffs, _, _, _ = np.linalg.lstsq(design, z_fit, rcond=None)
                coeff_update[:, iy, ix] = coeffs[:nbasis]

        return self._interpolate_missing_coefficient_images(coeff_update)

    def _residual_offset_terms(self, dx, dy, ix, iy, nx, ny):
        """
        Return the within-cell offset terms used in the residual fit for
        grid point ``(iy, ix)``.
        """
        offset_terms = [dx, dy]
        xcenter = nx // 2
        ycenter = ny // 2
        quadratic_offset_core_size = self._auto_quadratic_offset_core_size(
            (ny, nx))
        half_core_x = quadratic_offset_core_size[1] // 2
        half_core_y = quadratic_offset_core_size[0] // 2
        in_core = (abs(ix - xcenter) <= half_core_x
                   and abs(iy - ycenter) <= half_core_y)
        if in_core:
            offset_terms.extend((dx**2, dx * dy, dy**2))
        return offset_terms

    def _compute_residual_model_mismatch(self, residuals, det_x, det_y,
                                         x_coords, y_coords):
        """
        Compute the per-sample mismatch between the residual values and
        the fitted residual model.
        """
        nbasis = self._n_basis()
        basis = SpatialEPSFModel(np.zeros((nbasis, 3, 3)),
                                 oversampling=self.oversampling,
                                 detector_shape=self.detector_shape,
                                 degree=self.degree,
                                 detector_origin=self.detector_origin,
                                 detector_span=self.detector_span,
                                 epsf_class=self.epsf_class).basis_vector(
                                     det_x, det_y)
        _, ny, nx = residuals.shape
        mismatch = np.full_like(residuals, np.nan, dtype=float)

        for iy in range(ny):
            for ix in range(nx):
                z = residuals[:, iy, ix]
                dx = x_coords[:, iy, ix]
                dy = y_coords[:, iy, ix]
                valid = (np.isfinite(z) & np.isfinite(dx) & np.isfinite(dy))
                if np.count_nonzero(valid) < self.residual_min_valid_samples:
                    continue

                z_valid = z[valid]
                dx_valid = dx[valid]
                dy_valid = dy[valid]
                design = basis[:, valid].T
                design = np.column_stack(
                    (design, *self._residual_offset_terms(dx_valid, dy_valid,
                                                          ix, iy, nx, ny)))

                if self._sigma_clip is not None:
                    clipped = self._sigma_clip(z_valid, axis=0, masked=True,
                                               return_bounds=False)
                    keep = ~clipped.mask
                    z_fit = clipped.data[keep]
                    design_fit = design[keep]
                else:
                    z_fit = z_valid
                    design_fit = design
                if z_fit.size < self.residual_min_valid_samples:
                    continue

                coeffs, _, _, _ = np.linalg.lstsq(design_fit, z_fit, rcond=None)
                pred = design @ coeffs
                mismatch_indices = np.flatnonzero(valid)
                mismatch[mismatch_indices, iy, ix] = z_valid - pred

        return mismatch

    def _smooth_coefficients(self, coeff_data):
        if self._smooth_kernel is None:
            return coeff_data
        result = np.empty_like(coeff_data)
        for i in range(coeff_data.shape[0]):
            result[i] = convolve(coeff_data[i], self._smooth_kernel)
        return result

    def _shift_oversampled_image(self, data, *, dx, dy, origin=None,
                                 fill_value=0.0):
        """
        Shift an oversampled image by ``(dx, dy)`` detector pixels.
        """
        image_psf = self.epsf_class(data=np.asanyarray(data, dtype=float),
                                    oversampling=self.oversampling,
                                    origin=origin, fill_value=fill_value)
        y, x = np.indices(image_psf.data.shape, dtype=float)
        x /= image_psf.oversampling[1]
        y /= image_psf.oversampling[0]
        # Evaluate on the native oversampled grid; this must be identity
        # when dx=dy=0, so include the origin baseline in x_0/y_0.
        x0 = image_psf.origin[0] / image_psf.oversampling[1] - dx
        y0 = image_psf.origin[1] / image_psf.oversampling[0] - dy
        return image_psf.evaluate(x=x, y=y, flux=1.0, x_0=x0, y_0=y0)

    def _measure_recentering_shift(self, coeff_data, sample_positions):
        """
        Measure a common recentering shift from a representative local
        ePSF built at the median sampled detector position.
        """
        if len(sample_positions) == 0:
            return 0.0, 0.0

        sample_positions = np.asanyarray(sample_positions, dtype=float)
        xref = float(np.nanmedian(sample_positions[:, 0]))
        yref = float(np.nanmedian(sample_positions[:, 1]))
        spatial_model = SpatialEPSFModel(
            coeff_data, oversampling=self.oversampling,
            detector_shape=self.detector_shape, degree=self.degree,
            origin=None, fill_value=0.0,
            detector_origin=self.detector_origin,
            detector_span=self.detector_span,
            normalize_local_epsf=self.normalise_epsf,
            epsf_class=self.epsf_class)
        local_epsf = spatial_model.make_image_psf(xref, yref)
        epsf_data = np.array(local_epsf.data, copy=True)

        box_size = np.rint((np.asarray(epsf_data.shape, dtype=float) - 1.0)
                           / np.asarray(self.oversampling,
                                        dtype=float)).astype(int)
        box_size = np.maximum(box_size, 3)
        box_size = np.where(box_size % 2 == 0, box_size + 1, box_size)

        xcenter, ycenter = local_epsf.origin
        y, x = np.indices(epsf_data.shape, dtype=float)
        x /= self.oversampling[1]
        y /= self.oversampling[0]

        dx_total = 0.0
        dy_total = 0.0
        maxiters = 10
        center_dist_sq = self.center_accuracy_sq + 1.0e6
        center_dist_sq_prev = center_dist_sq + 1.0
        iter_num = 0

        while iter_num < maxiters and center_dist_sq >= self.center_accuracy_sq:
            iter_num += 1
            slices_large, _ = overlap_slices(
                epsf_data.shape, box_size * self.oversampling,
                (ycenter, xcenter))
            epsf_cutout = epsf_data[slices_large]
            mask = ~np.isfinite(epsf_cutout)
            xcenter_new, ycenter_new = centroid_com(epsf_cutout, mask=mask)
            xcenter_new += slices_large[1].start
            ycenter_new += slices_large[0].start

            dx = (xcenter_new - xcenter) / self.oversampling[1]
            dy = (ycenter_new - ycenter) / self.oversampling[0]
            center_dist_sq = dx**2 + dy**2
            if center_dist_sq >= center_dist_sq_prev:
                break
            center_dist_sq_prev = center_dist_sq

            dx_total += dx
            dy_total += dy
            new_x_0 = (xcenter / self.oversampling[1]) - dx_total
            new_y_0 = (ycenter / self.oversampling[0]) - dy_total
            epsf_data = local_epsf.evaluate(x=x, y=y, flux=1.0,
                                            x_0=new_x_0, y_0=new_y_0)

        return dx_total, dy_total

    def _recenter_coefficients(self, coeff_data, sample_positions):
        """
        Apply a common recentering shift to every spatial coefficient
        image.
        """
        dx, dy = self._measure_recentering_shift(coeff_data, sample_positions)
        if dx == 0.0 and dy == 0.0:
            return coeff_data

        result = np.empty_like(coeff_data)
        for i in range(coeff_data.shape[0]):
            result[i] = self._shift_oversampled_image(coeff_data[i], dx=dx,
                                                      dy=dy)
        return result

    def _normalise_coefficients(self, coeff_data, sample_positions):
        if not self.normalise_epsf or len(sample_positions) == 0:
            return coeff_data

        sample_positions = np.asanyarray(sample_positions, dtype=float)
        xref = float(np.nanmedian(sample_positions[:, 0]))
        yref = float(np.nanmedian(sample_positions[:, 1]))
        model = SpatialEPSFModel(coeff_data, oversampling=self.oversampling,
                                 detector_shape=self.detector_shape,
                                 degree=self.degree,
                                 detector_origin=self.detector_origin,
                                 detector_span=self.detector_span,
                                 normalize_local_epsf=False,
                                 epsf_class=self.epsf_class)
        local_data = model.local_epsf_data(xref, yref)
        scale = model._local_epsf_normalization(local_data)
        if np.isfinite(scale) and scale > 0.0:
            coeff_data = coeff_data / scale
        return coeff_data

    def _collect_ppe_residual_results(self, stars):
        """
        Collect linked-star residual samples for PPE diagnostics.
        """
        residual_results = {'subpixel_x': [], 'subpixel_y': [],
                            'x_residual': [], 'y_residual': [],
                            'flux_residual': [], 'det_x': [], 'det_y': []}

        for linked_star in stars:
            if not isinstance(linked_star, LinkedEPSFStar):
                continue

            mean_flux = linked_star.get_mean_flux()
            mean_radec = linked_star.get_mean_radec()
            if mean_flux is None or mean_radec is None:
                continue
            mean_ra, mean_dec = mean_radec
            if not np.isfinite(mean_flux):
                continue

            for star in linked_star.all_good_stars:
                if star.wcs_large is None:
                    continue
                mean_x, mean_y = star.wcs_large.world_to_pixel_values(
                    mean_ra, mean_dec)
                if not (np.isfinite(mean_x) and np.isfinite(mean_y)):
                    continue

                residual_results['subpixel_x'].append(np.mod(star.center[0],
                                                             1.0))
                residual_results['subpixel_y'].append(np.mod(star.center[1],
                                                             1.0))
                residual_results['x_residual'].append(star.center[0] - mean_x)
                residual_results['y_residual'].append(star.center[1] - mean_y)
                residual_results['det_x'].append(star.center[0])
                residual_results['det_y'].append(star.center[1])
                if mean_flux == 0.0:
                    residual_results['flux_residual'].append(0.0)
                else:
                    residual_results['flux_residual'].append(
                        (star.flux - mean_flux) / mean_flux)

        return residual_results

    def _collect_detector_position_results(self, stars, *, linked_only=False):
        """
        Collect detector-position samples from input stars.

        Parameters
        ----------
        stars : `EPSFStars`
            The input stars container.

        linked_only : bool, optional
            If `True`, include only members of `LinkedEPSFStar` objects.

        Returns
        -------
        results : dict
            A dictionary containing detector positions and linked-group
            labels for the collected star samples.
        """
        results = {'det_x': [], 'det_y': [], 'group_id': [],
                   'group_mean_x': [], 'group_mean_y': []}

        group_id = 0
        for item in stars:
            if isinstance(item, LinkedEPSFStar):
                good_stars = item.all_good_stars
                if len(good_stars) == 0:
                    group_id += 1
                    continue

                xvals = np.asarray([star.center[0] for star in good_stars],
                                   dtype=float)
                yvals = np.asarray([star.center[1] for star in good_stars],
                                   dtype=float)
                mean_x = np.nanmean(xvals)
                mean_y = np.nanmean(yvals)

                for star in good_stars:
                    results['det_x'].append(star.center[0])
                    results['det_y'].append(star.center[1])
                    results['group_id'].append(group_id)
                    results['group_mean_x'].append(mean_x)
                    results['group_mean_y'].append(mean_y)
                group_id += 1
                continue

            if linked_only or item._excluded_from_fit:
                continue

            results['det_x'].append(item.center[0])
            results['det_y'].append(item.center[1])
            results['group_id'].append(-1)
            results['group_mean_x'].append(item.center[0])
            results['group_mean_y'].append(item.center[1])

        for key in ('det_x', 'det_y', 'group_id',
                    'group_mean_x', 'group_mean_y'):
            results[key] = np.asarray(results[key])

        return results

    def _collect_residual_sample_metadata(self, stars):
        """
        Collect per-sample metadata aligned with ``stars.all_good_stars``.

        Returns
        -------
        metadata : dict
            Dictionary containing linked-group ids, linked-group mean
            fluxes, and frame IDs for each good sample.
        """
        group_id = []
        group_flux = []
        frame_id = []

        linked_id = 0
        for item in stars:
            if isinstance(item, LinkedEPSFStar):
                good_stars = item.all_good_stars
                if len(good_stars) == 0:
                    linked_id += 1
                    continue

                mean_flux = np.nanmean([star.flux for star in good_stars])
                for star in good_stars:
                    group_id.append(linked_id)
                    group_flux.append(mean_flux)
                    frame_id.append(star.frame_id)
                linked_id += 1
                continue

            if item._excluded_from_fit:
                continue
            group_id.append(-1)
            group_flux.append(item.flux)
            frame_id.append(item.frame_id)

        return {
            'group_id': np.asarray(group_id, dtype=int),
            'group_flux': np.asarray(group_flux, dtype=float),
            'frame_id': np.asarray(frame_id, dtype=object),
        }

    def plot_sample_distribution(self, stars, *, bins=25, linked_only=False,
                                 show_group_means=True):
        """
        Plot the detector-position distribution of the sample stars.

        Parameters
        ----------
        stars : `EPSFStars`
            The input stars container.

        bins : int or tuple of int, optional
            Number of 2D histogram bins in detector ``(y, x)`` order. The
            default is 25.

        linked_only : bool, optional
            If `True`, include only members of `LinkedEPSFStar` objects.

        show_group_means : bool, optional
            If `True`, overplot one marker per linked group at its mean
            detector position.

        Returns
        -------
        fig, axes : tuple
            The matplotlib figure and axes array.
        """
        import matplotlib.pyplot as plt

        results = self._collect_detector_position_results(
            stars, linked_only=linked_only)
        det_x = results['det_x']
        det_y = results['det_y']
        group_id = results['group_id']
        group_mean_x = results['group_mean_x']
        group_mean_y = results['group_mean_y']

        if det_x.size == 0 or det_y.size == 0:
            raise ValueError('No valid detector-position samples were found')

        bins = as_pair('bins', bins, lower_bound=(1, 1))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                                 constrained_layout=True)

        axes[0].scatter(det_x, det_y, s=10, alpha=0.6, color='tab:blue')
        if show_group_means and np.any(group_id >= 0):
            unique_groups = np.unique(group_id[group_id >= 0])
            mean_x = [group_mean_x[group_id == gid][0] for gid in unique_groups]
            mean_y = [group_mean_y[group_id == gid][0] for gid in unique_groups]
            axes[0].scatter(mean_x, mean_y, s=50, marker='x',
                            color='tab:red', label='Linked-star means')
            axes[0].legend(loc='best')
        axes[0].set_title('Detector Sample Positions')
        axes[0].set_xlabel('Detector X')
        axes[0].set_ylabel('Detector Y')
        axes[0].set_aspect('equal', adjustable='box')

        hist = axes[1].hist2d(det_x, det_y, bins=(bins[1], bins[0]),
                              cmap='viridis')
        axes[1].set_title('2D Sample Density')
        axes[1].set_xlabel('Detector X')
        axes[1].set_ylabel('Detector Y')
        axes[1].set_aspect('equal', adjustable='box')
        fig.colorbar(hist[3], ax=axes[1], label='Samples per bin')

        xmin, xmax = np.nanmin(det_x), np.nanmax(det_x)
        ymin, ymax = np.nanmin(det_y), np.nanmax(det_y)
        xedges = np.linspace(xmin, xmax, 4)
        yedges = np.linspace(ymin, ymax, 4)
        region_counts = np.zeros((3, 3), dtype=int)
        for iy in range(3):
            for ix in range(3):
                if ix == 2:
                    xmask = (det_x >= xedges[ix]) & (det_x <= xedges[ix + 1])
                else:
                    xmask = (det_x >= xedges[ix]) & (det_x < xedges[ix + 1])
                if iy == 2:
                    ymask = (det_y >= yedges[iy]) & (det_y <= yedges[iy + 1])
                else:
                    ymask = (det_y >= yedges[iy]) & (det_y < yedges[iy + 1])
                region_counts[iy, ix] = np.count_nonzero(xmask & ymask)

        im = axes[2].imshow(region_counts, origin='lower', cmap='magma')
        axes[2].set_title('3x3 Region Counts')
        axes[2].set_xlabel('X Region')
        axes[2].set_ylabel('Y Region')
        for iy in range(3):
            for ix in range(3):
                axes[2].text(ix, iy, str(region_counts[iy, ix]),
                             ha='center', va='center', color='white')
        fig.colorbar(im, ax=axes[2], label='Samples per region')

        plt.show()
        return fig, axes

    def plot_subpixel_distribution(self, stars, *, bins=None):
        """
        Plot the sub-pixel sampling distribution of the input stars.

        This shows:
        1. the overall detector sub-pixel phases,
        2. the sub-pixel shifts within each linked star relative to that
           linked star's reference phase,
        3. the number of linked stars contributing to each sub-pixel
           grid cell.

        Parameters
        ----------
        stars : `EPSFStars`
            The input stars container.

        bins : int or tuple of int, optional
            Number of sub-pixel bins in ``(y, x)`` order. If `None`, then
            ``self.ppe_supersampling`` is used.

        Returns
        -------
        fig, axes : tuple
            The matplotlib figure and axes array.
        """
        import matplotlib.pyplot as plt

        if bins is None:
            bins = self.ppe_supersampling
        bins = as_pair('bins', bins, lower_bound=(1, 1))

        overall_x = []
        overall_y = []
        shift_x = []
        shift_y = []
        overall_group_id = []
        next_group_id = 0

        for item in stars:
            if isinstance(item, LinkedEPSFStar):
                good_stars = item.all_good_stars
                if len(good_stars) == 0:
                    continue

                ref_x = np.mod(good_stars[0].center[0], 1.0)
                ref_y = np.mod(good_stars[0].center[1], 1.0)

                for star in good_stars:
                    phase_x = np.mod(star.center[0], 1.0)
                    phase_y = np.mod(star.center[1], 1.0)
                    overall_x.append(phase_x)
                    overall_y.append(phase_y)
                    overall_group_id.append(next_group_id)
                    shift_x.append(np.mod(phase_x - ref_x, 1.0))
                    shift_y.append(np.mod(phase_y - ref_y, 1.0))
                next_group_id += 1
                continue

            if item._excluded_from_fit:
                continue
            overall_x.append(np.mod(item.center[0], 1.0))
            overall_y.append(np.mod(item.center[1], 1.0))
            overall_group_id.append(next_group_id)
            next_group_id += 1

        overall_x = np.asarray(overall_x, dtype=float)
        overall_y = np.asarray(overall_y, dtype=float)
        shift_x = np.asarray(shift_x, dtype=float)
        shift_y = np.asarray(shift_y, dtype=float)
        overall_group_id = np.asarray(overall_group_id, dtype=int)

        if overall_x.size == 0 or overall_y.size == 0:
            raise ValueError('No valid sub-pixel samples were found')

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                                 constrained_layout=True)

        axes[0].scatter(overall_x, overall_y, s=10, alpha=0.6,
                        color='tab:blue')
        axes[0].set_title('Overall Sub-pixel Distribution')
        axes[0].set_xlabel('Sub-pixel x phase')
        axes[0].set_ylabel('Sub-pixel y phase')
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_aspect('equal', adjustable='box')

        if shift_x.size > 0 and shift_y.size > 0:
            axes[1].scatter(shift_x, shift_y, s=10, alpha=0.6,
                            color='tab:green')
        axes[1].set_title('Relative Sub-pixel Shifts')
        axes[1].set_xlabel('Shift x phase')
        axes[1].set_ylabel('Shift y phase')
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_aspect('equal', adjustable='box')

        unique_group_counts = np.zeros((bins[0], bins[1]), dtype=int)
        xbin = np.floor(overall_x * bins[1]).astype(int)
        ybin = np.floor(overall_y * bins[0]).astype(int)
        xbin = np.clip(xbin, 0, bins[1] - 1)
        ybin = np.clip(ybin, 0, bins[0] - 1)
        for j in range(bins[0]):
            for i in range(bins[1]):
                mask = (xbin == i) & (ybin == j)
                if np.any(mask):
                    unique_group_counts[j, i] = len(
                        np.unique(overall_group_id[mask]))

        im = axes[2].imshow(unique_group_counts, origin='lower',
                            extent=(0.0, 1.0, 0.0, 1.0),
                            cmap='magma', aspect='equal')
        axes[2].set_title('Linked Stars per Sub-pixel Cell')
        axes[2].set_xlabel('Sub-pixel x phase')
        axes[2].set_ylabel('Sub-pixel y phase')
        axes[2].set_aspect('equal', adjustable='box')
        fig.colorbar(im, ax=axes[2], label='Linked stars per cell')

        plt.show()
        return fig, axes

    def _generate_ppe_map(self, stars, supersampling=None):
        """
        Generate a global PPE map from fitted linked stars.
        """
        if supersampling is None:
            supersampling = self.oversampling
        supersampling = as_pair('supersampling', supersampling,
                                lower_bound=(1, 1))

        ppe_flux_map = np.full((supersampling[0], supersampling[1]), np.nan)
        ppe_x_map = np.full((supersampling[0], supersampling[1]), np.nan)
        ppe_y_map = np.full((supersampling[0], supersampling[1]), np.nan)

        residual_results = self._collect_ppe_residual_results(stars)
        if len(residual_results['subpixel_x']) == 0:
            return PPEMap(supersampling, np.zeros_like(ppe_flux_map),
                          np.zeros_like(ppe_x_map),
                          np.zeros_like(ppe_y_map))

        subpixel_x = np.asarray(residual_results['subpixel_x'])
        subpixel_y = np.asarray(residual_results['subpixel_y'])
        flux_residual = np.asarray(residual_results['flux_residual'])
        x_residual = np.asarray(residual_results['x_residual'])
        y_residual = np.asarray(residual_results['y_residual'])

        xbin = np.floor(subpixel_x * supersampling[1]).astype(int)
        ybin = np.floor(subpixel_y * supersampling[0]).astype(int)
        xbin = np.clip(xbin, 0, supersampling[1] - 1)
        ybin = np.clip(ybin, 0, supersampling[0] - 1)

        for j in range(supersampling[0]):
            for i in range(supersampling[1]):
                mask = (xbin == i) & (ybin == j)
                if not np.any(mask):
                    continue
                ppe_flux_map[j, i] = np.nanmedian(flux_residual[mask])
                ppe_x_map[j, i] = np.nanmedian(x_residual[mask])
                ppe_y_map[j, i] = np.nanmedian(y_residual[mask])

        for ppe_map, fill_value in ((ppe_flux_map, 0.0), (ppe_x_map, 0.0),
                                    (ppe_y_map, 0.0)):
            mask = ~np.isfinite(ppe_map)
            if not np.any(mask):
                continue
            if np.all(mask):
                ppe_map[:] = fill_value
                continue
            ppe_map[:] = _interpolate_missing_data(ppe_map, mask=mask,
                                                   method='nearest')
            ppe_map[~np.isfinite(ppe_map)] = fill_value

        return PPEMap(supersampling, ppe_flux_map, ppe_x_map, ppe_y_map)

    def _fit_ppe_coefficients(self, values, subpixel_x, subpixel_y, det_x,
                              det_y):
        nbasis = len(SpatialEPSFModel._basis_labels_for_degree(
            self.ppe_degree))
        coeff_data = np.zeros((nbasis, self.ppe_supersampling[0],
                               self.ppe_supersampling[1]), dtype=float)
        basis = SpatialEPSFModel(np.zeros((nbasis, 3, 3)),
                                 oversampling=(1, 1),
                                 detector_shape=self.detector_shape,
                                 degree=self.ppe_degree,
                                 detector_origin=self.detector_origin,
                                 detector_span=self.detector_span,
                                 epsf_class=self.epsf_class).basis_vector(
                                     det_x, det_y)

        xbin = np.floor(subpixel_x * self.ppe_supersampling[1]).astype(int)
        ybin = np.floor(subpixel_y * self.ppe_supersampling[0]).astype(int)
        xbin = np.clip(xbin, 0, self.ppe_supersampling[1] - 1)
        ybin = np.clip(ybin, 0, self.ppe_supersampling[0] - 1)

        for j in range(self.ppe_supersampling[0]):
            for i in range(self.ppe_supersampling[1]):
                mask = ((xbin == i) & (ybin == j)
                        & np.isfinite(values))
                if np.count_nonzero(mask) < self.ppe_min_valid_samples:
                    continue

                z_valid = values[mask]
                design = basis[:, mask].T
                if self._sigma_clip is not None:
                    clipped = self._sigma_clip(z_valid, axis=0, masked=True,
                                               return_bounds=False)
                    keep = ~clipped.mask
                    z_fit = clipped.data[keep]
                    design = design[keep]
                else:
                    z_fit = z_valid
                if z_fit.size < self.ppe_min_valid_samples:
                    continue

                coeffs, _, _, _ = np.linalg.lstsq(design, z_fit, rcond=None)
                coeff_data[:, j, i] = coeffs

        return coeff_data

    def _fit_spatial_ppe_model(self, stars):
        """
        Fit detector-position-dependent PPE maps from raw fitted stars.
        """
        if len(self.calibrate_ppe) == 0:
            return self._make_empty_ppe_model()

        residual_results = self._collect_ppe_residual_results(stars)
        if len(residual_results['subpixel_x']) == 0:
            return self._make_empty_ppe_model()

        subpixel_x = np.asarray(residual_results['subpixel_x'], dtype=float)
        subpixel_y = np.asarray(residual_results['subpixel_y'], dtype=float)
        det_x = np.asarray(residual_results['det_x'], dtype=float)
        det_y = np.asarray(residual_results['det_y'], dtype=float)

        flux_coeff = self._fit_ppe_coefficients(
            np.asarray(residual_results['flux_residual'], dtype=float),
            subpixel_x, subpixel_y, det_x, det_y)
        x_coeff = self._fit_ppe_coefficients(
            np.asarray(residual_results['x_residual'], dtype=float),
            subpixel_x, subpixel_y, det_x, det_y)
        y_coeff = self._fit_ppe_coefficients(
            np.asarray(residual_results['y_residual'], dtype=float),
            subpixel_x, subpixel_y, det_x, det_y)

        return SpatialPPEMapModel(flux_coeff, x_coeff, y_coeff,
                                  supersampling=self.ppe_supersampling,
                                  detector_shape=self.detector_shape,
                                  degree=self.ppe_degree,
                                  detector_origin=self.detector_origin,
                                  detector_span=self.detector_span)

    def _apply_spatial_ppe_corrections(self, stars, spatial_ppe_model):
        """
        Apply spatially varying PPE corrections to fitted stars.
        """
        apply_flux = 'Flux' in self.calibrate_ppe
        apply_position = 'Position' in self.calibrate_ppe
        if not apply_flux and not apply_position:
            return copy.deepcopy(stars)

        corrected = copy.deepcopy(stars)
        for star in corrected.all_stars:
            corrected_flux, corrected_x, corrected_y = spatial_ppe_model(
                star.flux, star.center[0], star.center[1],
                det_x=star.center[0], det_y=star.center[1])
            if apply_flux:
                star.flux = corrected_flux
            if apply_position:
                star.cutout_center = np.array((corrected_x, corrected_y),
                                              dtype=float) - star.origin

        return corrected

    def _plot_iteration_diagnostics(self, spatial_model, coeff_update,
                                    ppe_stars_before, ppe_stars_after,
                                    det_x, det_y, iter_num):
        """
        Plot three diagnostic figures:
        1. local ePSF models across a 3x3 grid spanning the sampled
           detector region,
        2. local residual-update images across the same 3x3 grid,
        3. overall PPE maps before and after the spatial PPE correction.
        """
        if not self.plot_diagnostics:
            return

        import matplotlib.pyplot as plt

        def _grid_positions():
            if len(det_x) == 0 or len(det_y) == 0:
                xvals = np.linspace(0.0, self.detector_shape[1] - 1, 3)
                yvals = np.linspace(0.0, self.detector_shape[0] - 1, 3)
            else:
                xmin = np.nanmin(det_x)
                xmax = np.nanmax(det_x)
                ymin = np.nanmin(det_y)
                ymax = np.nanmax(det_y)
                if not np.isfinite([xmin, xmax, ymin, ymax]).all():
                    xvals = np.linspace(0.0, self.detector_shape[1] - 1, 3)
                    yvals = np.linspace(0.0, self.detector_shape[0] - 1, 3)
                else:
                    if np.isclose(xmin, xmax):
                        xvals = np.repeat(xmin, 3)
                    else:
                        xvals = np.linspace(xmin, xmax, 3)
                    if np.isclose(ymin, ymax):
                        yvals = np.repeat(ymin, 3)
                    else:
                        yvals = np.linspace(ymin, ymax, 3)
            xx, yy = np.meshgrid(xvals, yvals)
            return list(zip(xx.ravel(), yy.ravel(), strict=True))

        grid_positions = _grid_positions()
        residual_model = SpatialEPSFModel(
            coeff_update, oversampling=self.oversampling,
            detector_shape=self.detector_shape, degree=self.degree,
            detector_origin=self.detector_origin,
            detector_span=self.detector_span,
            normalize_local_epsf=self.normalise_epsf,
            epsf_class=self.epsf_class)

        fig_model, axes_model = plt.subplots(3, 3, figsize=(12, 11),
                                             constrained_layout=True)
        for ax, (xpos, ypos) in zip(axes_model.ravel(), grid_positions):
            model_img = spatial_model.local_epsf_data(xpos, ypos)
            im = ax.imshow(model_img, origin='lower', cmap='viridis')
            ax.set_title(f'X={xpos:.1f}, Y={ypos:.1f}')
            ax.set_xlabel('Oversampled X')
            ax.set_ylabel('Oversampled Y')
            fig_model.colorbar(im, ax=ax)
        fig_model.suptitle(f'Iteration {iter_num}: Local ePSF Models',
                           fontsize=14)

        fig_resid, axes_resid = plt.subplots(3, 3, figsize=(12, 11),
                                             constrained_layout=True)
        for ax, (xpos, ypos) in zip(axes_resid.ravel(), grid_positions):
            residual_img = residual_model.local_epsf_data(xpos, ypos)
            im = ax.imshow(residual_img, origin='lower', cmap='coolwarm')
            ax.set_title(f'X={xpos:.1f}, Y={ypos:.1f}')
            ax.set_xlabel('Oversampled X')
            ax.set_ylabel('Oversampled Y')
            fig_resid.colorbar(im, ax=ax)
        fig_resid.suptitle(f'Iteration {iter_num}: Residual Updates',
                           fontsize=14)

        sample_factor = 3
        extent = (0.0, 1.0, 0.0, 1.0)

        def _sample_ppe_maps(stars):
            ppe_map = self._generate_ppe_map(stars)
            yphase = np.linspace(0.0, 1.0,
                                 ppe_map.supersampling[0] * sample_factor,
                                 endpoint=False)
            xphase = np.linspace(0.0, 1.0,
                                 ppe_map.supersampling[1] * sample_factor,
                                 endpoint=False)
            xx, yy = np.meshgrid(xphase, yphase)
            return (
                ppe_map._sample_periodic_map(ppe_map.flux_ppe, xx, yy,
                                             ppe_map.supersampling),
                ppe_map._sample_periodic_map(ppe_map.x_ppe, xx, yy,
                                             ppe_map.supersampling),
                ppe_map._sample_periodic_map(ppe_map.y_ppe, xx, yy,
                                             ppe_map.supersampling),
            )

        ppe_before = _sample_ppe_maps(ppe_stars_before)
        ppe_after = _sample_ppe_maps(ppe_stars_after)
        titles = (
            ('Flux PPE Before', 'Flux PPE After',
             'Fractional flux residual'),
            ('X PPE Before', 'X PPE After', 'X residual (pixels)'),
            ('Y PPE Before', 'Y PPE After', 'Y residual (pixels)'),
        )

        fig_ppe, axes_ppe = plt.subplots(2, 3, figsize=(14, 8),
                                         constrained_layout=True)
        for col, ((before_data, after_data), (before_title, after_title,
                                               cbar_label)) in enumerate(
                                                   zip(zip(ppe_before,
                                                           ppe_after),
                                                       titles),
                                                   start=0):
            im_before = axes_ppe[0, col].imshow(before_data, origin='lower',
                                                extent=extent,
                                                cmap='coolwarm',
                                                aspect='equal')
            axes_ppe[0, col].set_title(
                f'Iteration {iter_num}: {before_title}')
            axes_ppe[0, col].set_xlabel('Subpixel x phase')
            axes_ppe[0, col].set_ylabel('Subpixel y phase')
            fig_ppe.colorbar(im_before, ax=axes_ppe[0, col],
                             label=cbar_label)

            im_after = axes_ppe[1, col].imshow(after_data, origin='lower',
                                               extent=extent,
                                               cmap='coolwarm',
                                               aspect='equal')
            axes_ppe[1, col].set_title(
                f'Iteration {iter_num}: {after_title}')
            axes_ppe[1, col].set_xlabel('Subpixel x phase')
            axes_ppe[1, col].set_ylabel('Subpixel y phase')
            fig_ppe.colorbar(im_after, ax=axes_ppe[1, col],
                             label=cbar_label)

        fig_ppe.suptitle(f'Iteration {iter_num}: PPE Diagnostics',
                         fontsize=14)

        plt.show()

    def _plot_central_residual_fit(self, residuals, det_x, det_y, group_id,
                                   x_coords, y_coords, coeff_update,
                                   iter_num):
        """
        Plot the residual samples and fitted detector-position surface
        for the central oversampled grid point.

        The residual samples are color-coded by their squared distance
        from the center of the subpixel grid section, ``dx**2 + dy**2``.
        """
        if not self.plot_diagnostics or residuals.size == 0:
            return

        import matplotlib.pyplot as plt

        iy = residuals.shape[1] // 2
        ix = residuals.shape[2] // 2
        z = residuals[:, iy, ix]
        dx = x_coords[:, iy, ix]
        dy = y_coords[:, iy, ix]
        valid = (np.isfinite(z) & np.isfinite(det_x) & np.isfinite(det_y)
                 & np.isfinite(group_id) & np.isfinite(dx) & np.isfinite(dy))
        if np.count_nonzero(valid) == 0:
            return

        xdata = det_x[valid]
        ydata = det_y[valid]
        zdata = z[valid]
        gid = group_id[valid].astype(int)
        dxdata = dx[valid]
        dydata = dy[valid]
        point_metric = dxdata**2 + dydata**2

        xmin = np.nanmin(xdata)
        xmax = np.nanmax(xdata)
        ymin = np.nanmin(ydata)
        ymax = np.nanmax(ydata)
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            return

        xgrid = np.linspace(xmin, xmax, 25)
        ygrid = np.linspace(ymin, ymax, 25)
        yy, xx = np.meshgrid(ygrid, xgrid, indexing='ij')

        basis_model = SpatialEPSFModel(
            np.zeros((self._n_basis(), 3, 3)),
            oversampling=self.oversampling,
            detector_shape=self.detector_shape,
            degree=self.degree,
            detector_origin=self.detector_origin,
            detector_span=self.detector_span,
            normalize_local_epsf=self.normalise_epsf,
            epsf_class=self.epsf_class)
        sample_basis = basis_model.basis_vector(xdata, ydata).T
        surface_basis = basis_model.basis_vector(xx, yy)

        xcenter = residuals.shape[2] // 2
        ycenter = residuals.shape[1] // 2
        quadratic_offset_core_size = self._auto_quadratic_offset_core_size(
            residuals.shape[1:])
        half_core_x = quadratic_offset_core_size[1] // 2
        half_core_y = quadratic_offset_core_size[0] // 2
        in_core = (abs(ix - xcenter) <= half_core_x
                   and abs(iy - ycenter) <= half_core_y)

        design = sample_basis
        offset_terms = [dxdata, dydata]
        if in_core:
            offset_terms.extend((dxdata**2, dxdata * dydata, dydata**2))
        design = np.column_stack((design, *offset_terms))

        if self._sigma_clip is not None:
            clipped = self._sigma_clip(zdata, axis=0, masked=True,
                                       return_bounds=False)
            keep = ~clipped.mask
            z_fit = clipped.data[keep]
            design_fit = design[keep]
            xdata = xdata[keep]
            ydata = ydata[keep]
            zdata = zdata[keep]
            gid = gid[keep]
            dxdata = dxdata[keep]
            dydata = dydata[keep]
            point_metric = point_metric[keep]
        else:
            z_fit = zdata
            design_fit = design

        if z_fit.size < self.residual_min_valid_samples:
            return

        coeffs, _, _, _ = np.linalg.lstsq(design_fit, z_fit, rcond=None)
        detector_coeffs = coeffs[:self._n_basis()]
        offset_coeffs = coeffs[self._n_basis():]
        offset_design = np.column_stack(offset_terms)[keep]
        offset_contrib = np.sum(offset_design * offset_coeffs, axis=1)
        z_centered = zdata - offset_contrib
        surface = np.tensordot(detector_coeffs, surface_basis, axes=(0, 0))

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(121, projection='3d')
        scatter = ax.scatter(xdata, ydata, z_centered, s=20, alpha=0.85,
                             c=point_metric, cmap='plasma',
                             edgecolors='k', linewidths=0.2,
                             label='Residual samples')
        ax.plot_surface(xx, yy, surface, cmap='viridis', alpha=0.65,
                        linewidth=0, antialiased=True)
        ax.set_title('Iteration '
                     f'{iter_num}: Central Grid-point Residual Fit '
                     '(offset-corrected)')
        ax.set_xlabel('Detector X')
        ax.set_ylabel('Detector Y')
        ax.set_zlabel('Residual Value')
        ax.legend(loc='best')
        fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08,
                     label=r'$dx^2 + dy^2$')

        # Add a 2D offset panel showing the same color metric.
        inset = fig.add_axes([0.34, 0.62, 0.14, 0.22])
        inset.scatter(dxdata, dydata, c=point_metric, cmap='plasma', s=18,
                      edgecolors='k', linewidths=0.2)
        inset.set_title('Within-cell Offsets', fontsize=10)
        inset.set_xlabel('dx', fontsize=9)
        inset.set_ylabel('dy', fontsize=9)
        inset.set_xlim(-0.5, 0.5)
        inset.set_ylim(-0.5, 0.5)
        inset.tick_params(labelsize=8)
        inset.set_aspect('equal', adjustable='box')

        ax2 = fig.add_subplot(122)
        rr = np.array(z_centered, copy=True)
        linked_mask = gid >= 0
        for linked_idx in np.unique(gid[linked_mask]):
            mask = gid == linked_idx
            rr[mask] = z_centered[mask] - np.nanmean(z_centered[mask])
        ax2.scatter(gid, rr, c=point_metric, cmap='plasma', s=24, alpha=0.85,
                    edgecolors='k', linewidths=0.2)
        ax2.axhline(0.0, color='0.3', linestyle='--', linewidth=1)
        ax2.set_title('Per-star Residual of Residual')
        ax2.set_xlabel('Linked Star ID')
        ax2.set_ylabel('Residual - Linked-star Mean Residual')
        plt.show()

    def _plot_residual_scatter_diagnostics(self, residuals, det_x, det_y,
                                           group_id, x_coords, y_coords,
                                           iter_num):
        """
        Plot diagnostics of how well the fitted residual model matches
        the actual residual samples.
        """
        if not self.plot_diagnostics or residuals.size == 0:
            return

        import matplotlib.pyplot as plt

        mismatch = self._compute_residual_model_mismatch(
            residuals, det_x, det_y, x_coords, y_coords)
        sample_rms = np.sqrt(np.nanmean(mismatch**2, axis=(1, 2)))
        sample_mean = np.nanmean(mismatch, axis=(1, 2))
        valid = (np.isfinite(sample_rms) & np.isfinite(det_x)
                 & np.isfinite(det_y) & np.isfinite(group_id))
        if np.count_nonzero(valid) == 0:
            return

        group_id = group_id[valid]
        sample_rms = sample_rms[valid]
        sample_mean = sample_mean[valid]
        det_x = det_x[valid]
        det_y = det_y[valid]

        linked_mask = group_id >= 0
        fig, axes = plt.subplots(2, 3, figsize=(18, 5),
                                 constrained_layout=True)

        if np.count_nonzero(linked_mask) > 0:
            axes[0][0].scatter(group_id[linked_mask], sample_mean[linked_mask],
                            color='tab:orange', s=20, alpha=0.7)
            axes[1][0].scatter(group_id[linked_mask], sample_rms[linked_mask],
                            color='tab:orange', s=20, alpha=0.7)
            
        axes[0][0].set_title(
            f'Iteration {iter_num}: Residual-model Mean vs Linked Star')
        axes[0][0].set_xlabel('Linked star ID')
        axes[0][0].set_ylabel('Mean(actual residual - residual model)')

        axes[1][0].set_title(
            f'Iteration {iter_num}: Residual-model RMS vs Linked Star')
        axes[1][0].set_xlabel('Linked star ID')
        axes[1][0].set_ylabel('RMS(actual residual - residual model)')

        sc_mean = axes[0][1].scatter(det_x, det_y, c=sample_mean, s=24, alpha=0.8,
                             cmap='viridis')
        
        axes[0][1].set_ylabel('Detector Y')
        axes[0][1].set_xlabel('Detector X')
        axes[0][1].set_title(f'Iteration {iter_num}: Residual-model Mean vs Detector Position')
        axes[0][1].set_aspect('equal', adjustable='box')
        fig.colorbar(sc_mean, ax=axes[0][1],
                     label='Mean(actual residual - residual model)')
        
        sc_rms = axes[1][1].scatter(det_x, det_y, c=sample_rms, s=24, alpha=0.8,
                             cmap='viridis')
        
        axes[1][1].set_ylabel('Detector Y')
        axes[1][1].set_xlabel('Detector X')
        axes[1][1].set_title(f'Iteration {iter_num}: Residual-model RMS vs Detector Position')
        axes[1][1].set_aspect('equal', adjustable='box')
        fig.colorbar(sc_rms, ax=axes[1][1],
                     label='RMS(actual residual - residual model)')

        grid_mean = np.nanmean(mismatch, axis=0)
        grid_rms = np.sqrt(np.nanmean(mismatch**2, axis=0))
        im_mean = axes[0][2].imshow(grid_mean, origin='lower', cmap='magma')
        im_rms = axes[1][2].imshow(grid_rms, origin='lower', cmap='magma')
        axes[0][2].set_title(
            f'Iteration {iter_num}: Residual-model Mean per Grid Section')
        axes[0][2].set_xlabel('Oversampled X')
        axes[0][2].set_ylabel('Oversampled Y')
        fig.colorbar(im_mean, ax=axes[0][2],
                     label='Mean(actual residual - residual model)')
        axes[1][2].set_title(
            f'Iteration {iter_num}: Residual-model RMS per Grid Section')
        axes[1][2].set_xlabel('Oversampled X')
        axes[1][2].set_ylabel('Oversampled Y')
        fig.colorbar(im_rms, ax=axes[1][2],
                     label='RMS(actual residual - residual model)')

        plt.show()

        # Make a separate plot showing the residual-model mean and RMS per gridsection but split up into nxn regions across the area that the samples are in.
        n_regions = 2
        mean_fig, mean_axes = plt.subplots(n_regions, n_regions, figsize=(12, 10), constrained_layout=True)
        rms_fig, rms_axes = plt.subplots(n_regions, n_regions, figsize=(12, 10), constrained_layout=True)
        x_min = np.nanmin(det_x)
        x_max = np.nanmax(det_x)
        y_min = np.nanmin(det_y)
        y_max = np.nanmax(det_y)
        x_edges = np.linspace(x_min, x_max, n_regions + 1)
        y_edges = np.linspace(y_min, y_max, n_regions + 1)
        for i in range(n_regions):
            for j in range(n_regions):
                region_mask = (det_x >= x_edges[i]) & (det_x < x_edges[i + 1]) & (det_y >= y_edges[j]) & (det_y < y_edges[j + 1])
                if np.count_nonzero(region_mask) == 0:
                    continue
                grid_mean = np.nanmean(mismatch[region_mask], axis=0)
                grid_rms = np.sqrt(np.nanmean(mismatch[region_mask]**2, axis=0))
                im_mean = mean_axes[n_regions-1-j][i].imshow(grid_mean, origin='lower', cmap='magma')
                im_rms = rms_axes[n_regions-1-j][i].imshow(grid_rms, origin='lower', cmap='magma')
                mean_fig.colorbar(im_mean, ax=mean_axes[n_regions-1-j][i])
                rms_fig.colorbar(im_rms, ax=rms_axes[n_regions-1-j][i])
        mean_fig.suptitle(f'Iteration {iter_num}: Residual-model Mean per Grid Section (Split by Detector Region)', fontsize=14)
        rms_fig.suptitle(f'Iteration {iter_num}: Residual-model RMS per Grid Section (Split by Detector Region)', fontsize=14)
        plt.show()

    def build_epsf(self, stars, *, init_model=None):
        if not isinstance(stars, EPSFStars):
            raise TypeError('stars must be an EPSFStars object')

        self._models = []
        self._ppe_models = []
        self.final_ppe_model = None
        self._resolve_detector_geometry(stars, init_model=init_model)
        self._log('SpatialEPSFBuilder: creating initial spatial model')
        if self.plot_diagnostics:
            self._log('SpatialEPSFBuilder: plotting sample distribution')
            self.plot_sample_distribution(stars, linked_only=True)
            self._log('SpatialEPSFBuilder: plotting sub-pixel distribution')
            self.plot_subpixel_distribution(stars)
        spatial_model = (self._create_initial_model(stars)
                         if init_model is None else init_model.deepcopy())
        centers = stars.cutout_center_flat

        for iter_num in range(1, self.maxiters + 1):
            self._log(f'SpatialEPSFBuilder: iteration {iter_num}/{self.maxiters} '
                      'starting')
            spatial_model = copy.deepcopy(spatial_model)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} fitting stars')
            fitted_stars_raw = self.fitter(spatial_model, stars)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} fitting '
                      'spatial PPE correction surfaces')
            spatial_ppe_model = self._fit_spatial_ppe_model(fitted_stars_raw)
            self._ppe_models.append(spatial_ppe_model)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} applying '
                      'spatial PPE corrections to fitted stars')
            fitted_stars_corrected = self._apply_spatial_ppe_corrections(
                fitted_stars_raw, spatial_ppe_model)
            self.final_ppe_model = spatial_ppe_model

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} applying '
                      'linked-star constraints to PPE-corrected values')
            fitted_stars = self._apply_linked_constraints(
                fitted_stars_corrected)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} selecting '
                      'stars for residual stack')
            residual_stars = self._select_residual_stars(fitted_stars,
                                                         spatial_model)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} resampling '
                      'residual stack')
            residuals, weights, x_coords, y_coords, det_x, det_y, group_id = (
                self._resample_residuals(residual_stars, spatial_model))

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} '
                      'computing trust map from residual RMS')
            trust_map = self._compute_trust_map_from_residuals(
                residuals, weights)

            # Collapse star stack to a detector-position-dependent residual
            # coefficient update for each oversampled grid point.
            self._log(f'SpatialEPSFBuilder: iteration {iter_num} fitting '
                      'spatial residual coefficient surfaces')
            coeff_update = self._fit_residual_coefficients(
                residuals, det_x, det_y, x_coords=x_coords, y_coords=y_coords)

            if self._residual_smooth_kernel is not None:
                self._log(f'SpatialEPSFBuilder: iteration {iter_num} smoothing '
                          'residual coefficient surfaces')
                for ibasis in range(coeff_update.shape[0]):
                    coeff_update[ibasis] = convolve(coeff_update[ibasis],
                                                    self._residual_smooth_kernel)

            if iter_num % 10 == 0 or iter_num == self.maxiters:
                self._log(f'SpatialEPSFBuilder: iteration {iter_num} plotting '
                        'diagnostics')
                self._plot_iteration_diagnostics(spatial_model, coeff_update,
                                                fitted_stars_raw,
                                                fitted_stars_corrected,
                                                det_x, det_y, iter_num)
                self._plot_central_residual_fit(residuals, det_x, det_y,
                                                group_id, x_coords, y_coords,
                                                coeff_update, iter_num)
                self._plot_residual_scatter_diagnostics(
                    residuals, det_x, det_y, group_id, x_coords, y_coords,
                    iter_num)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} updating '
                      'spatial ePSF coefficients')
            coeff_data = spatial_model.coeff_data.copy()
            coeff_data += self.residual_update_fraction * coeff_update

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} smoothing '
                      'coefficient images')
            coeff_data = self._smooth_coefficients(coeff_data)

            sample_positions = list(zip(det_x, det_y, strict=True))

            if self.recenter_epsf:
                self._log(f'SpatialEPSFBuilder: iteration {iter_num} '
                          'recentering spatial ePSF')
                coeff_data = self._recenter_coefficients(coeff_data,
                                                         sample_positions)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} normalizing '
                      'spatial ePSF')
            coeff_data = self._normalise_coefficients(coeff_data,
                                                      sample_positions)

            self._log(f'SpatialEPSFBuilder: iteration {iter_num} rebuilding '
                      'spatial model object')
            spatial_model = SpatialEPSFModel(
                coeff_data, oversampling=self.oversampling,
                detector_shape=self.detector_shape, degree=self.degree,
                origin=spatial_model.origin,
                fill_value=spatial_model.fill_value,
                detector_origin=self.detector_origin,
                detector_span=self.detector_span,
                normalize_local_epsf=self.normalise_epsf,
                trust_map=trust_map,
                epsf_class=self.epsf_class)

            self._models.append(spatial_model)

            dx_dy = fitted_stars.cutout_center_flat - centers
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = fitted_stars.cutout_center_flat
            if (center_dist_sq.size > 0
                    and np.nanmax(center_dist_sq) < self.center_accuracy_sq):
                self._log(f'SpatialEPSFBuilder: converged after iteration '
                          f'{iter_num}')
                stars = fitted_stars
                break

            stars = fitted_stars

        self._log('SpatialEPSFBuilder: finished')

        return spatial_model, stars


class PPEMap:
    """
    Simple PPE correction map.

    This mirrors the interface used in ``epsf.py`` so the new module can
    still be used for PPE diagnostics if desired.
    """

    def __init__(self, supersampling, flux_ppe, x_ppe, y_ppe):
        self.supersampling = supersampling
        self.flux_ppe = flux_ppe
        self.x_ppe = x_ppe
        self.y_ppe = y_ppe

    @staticmethod
    def _sample_periodic_map(ppe_map, x, y, supersampling):
        xphase = np.mod(x, 1.0)
        yphase = np.mod(y, 1.0)
        xcoord = xphase * supersampling[1] - 0.5
        ycoord = yphase * supersampling[0] - 0.5
        x0 = np.floor(xcoord).astype(int)
        y0 = np.floor(ycoord).astype(int)
        dx = xcoord - x0
        dy = ycoord - y0
        x0 %= supersampling[1]
        y0 %= supersampling[0]
        x1 = (x0 + 1) % supersampling[1]
        y1 = (y0 + 1) % supersampling[0]

        return ((1.0 - dx) * (1.0 - dy) * ppe_map[y0, x0]
                + dx * (1.0 - dy) * ppe_map[y0, x1]
                + (1.0 - dx) * dy * ppe_map[y1, x0]
                + dx * dy * ppe_map[y1, x1])

    def __call__(self, flux, x, y):
        flux_scalar = np.isscalar(flux)
        x_scalar = np.isscalar(x)
        y_scalar = np.isscalar(y)

        flux, x, y = np.broadcast_arrays(np.asanyarray(flux, dtype=float),
                                         np.asanyarray(x, dtype=float),
                                         np.asanyarray(y, dtype=float))

        flux_residual = self._sample_periodic_map(self.flux_ppe, x, y,
                                                  self.supersampling)
        x_residual = self._sample_periodic_map(self.x_ppe, x, y,
                                               self.supersampling)
        y_residual = self._sample_periodic_map(self.y_ppe, x, y,
                                               self.supersampling)

        flux_scale = 1.0 + flux_residual
        invalid = (~np.isfinite(flux_scale)) | (flux_scale == 0.0)
        corrected_flux = np.array(flux, copy=True)
        np.divide(flux, flux_scale, out=corrected_flux, where=~invalid)
        corrected_x = x - np.where(np.isfinite(x_residual), x_residual, 0.0)
        corrected_y = y - np.where(np.isfinite(y_residual), y_residual, 0.0)

        if flux_scalar and x_scalar and y_scalar:
            return (corrected_flux.item(), corrected_x.item(),
                    corrected_y.item())
        return corrected_flux, corrected_x, corrected_y

    def plot_maps(self):
        """
        Plot the PPE maps for visual inspection.
        """
        import matplotlib.pyplot as plt

        sample_factor = 3
        yphase = np.linspace(0.0, 1.0, self.supersampling[0] * sample_factor,
                             endpoint=False)
        xphase = np.linspace(0.0, 1.0, self.supersampling[1] * sample_factor,
                             endpoint=False)
        xx, yy = np.meshgrid(xphase, yphase)

        flux_map = self._sample_periodic_map(self.flux_ppe, xx, yy,
                                             self.supersampling)
        x_map = self._sample_periodic_map(self.x_ppe, xx, yy,
                                          self.supersampling)
        y_map = self._sample_periodic_map(self.y_ppe, xx, yy,
                                          self.supersampling)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                 constrained_layout=True)
        map_data = (
            (flux_map, 'Flux PPE', 'Fractional flux residual'),
            (x_map, 'X PPE', 'X residual (pixels)'),
            (y_map, 'Y PPE', 'Y residual (pixels)'),
        )
        extent = (0.0, 1.0, 0.0, 1.0)

        for ax, (data, title, cbar_label) in zip(axes, map_data):
            im = ax.imshow(data, origin='lower', extent=extent,
                           cmap='coolwarm', aspect='equal')
            ax.set_title(title)
            ax.set_xlabel('Subpixel x phase')
            ax.set_ylabel('Subpixel y phase')
            fig.colorbar(im, ax=ax, label=cbar_label)

        return fig
