# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to build and fit an effective PSF (ePSF) based on Anderson
and King (2000; PASP 112, 1360) and Anderson (2016; WFC3 ISR 2016-12).
"""

import copy
from typing_extensions import final
import warnings

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve, label, median_filter

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF, _LegacyEPSFModel
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround
from photutils.utils._stats import nanmedian
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['EPSFBuilder', 'EPSFFitter', 'PPEMap']


class EPSFFitter:
    """
    Class to fit an ePSF model to one or more stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A `~astropy.modeling.fitting.Fitter` object, or
        `~photutils.psf.fitters.PriorLogTRFLSQFitter`. If `None`, then the
        default `~astropy.modeling.fitting.TRFLSQFitter` will be used.

    fit_boxsize : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be used
        for ePSF fitting. This allows using only a small number of
        central pixels of the star (i.e., where the star is brightest)
        for fitting. If ``fit_boxsize`` is a scalar then a square box of
        size ``fit_boxsize`` will be used. If ``fit_boxsize`` has two
        elements, they must be in ``(ny, nx)`` order. ``fit_boxsize``
        must have odd values and be greater than or equal to 3 for both
        axes. If `None`, the fitter will use the entire star image.

    **fitter_kwargs : dict, optional
        Any additional keyword arguments (except ``x``, ``y``, ``z``, or
        ``weights``) to be passed directly to the ``__call__()`` method
        of the input ``fitter``.
    """

    def __init__(self, *, fitter=None, fit_boxsize=3,
                 **fitter_kwargs):

        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        if fit_boxsize is not None:
            self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                    lower_bound=(3, 0), check_odd=True)
        else:
            self.fit_boxsize = fit_boxsize

        # remove any fitter keyword arguments that we need to set
        remove_kwargs = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            if kwarg in fitter_kwargs:
                del fitter_kwargs[kwarg]
        self.fitter_kwargs = fitter_kwargs

    def __call__(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit. The center coordinates for each star
            should be as close as possible to actual centers. For stars
            than contain weights, a weighted fit of the ePSF to the star
            will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars. The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, ImagePSF):
            msg = 'The input epsf must be an ImagePSF'
            raise TypeError(msg)

        # perform the fit
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                # make a copy of the input ePSF since the fitter will modify it
                _epsf = epsf.deepcopy()
                fitted_star = self._fit_star(_epsf, star, self.fitter,
                                             self.fitter_kwargs,
                                             self.fitter_has_fit_info,
                                             self.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    # make a copy of the input ePSF since the fitter will modify it
                    _epsf = epsf.deepcopy()
                    fitted_star.append(
                        self._fit_star(_epsf, linked_star, self.fitter,
                                       self.fitter_kwargs,
                                       self.fitter_has_fit_info,
                                       self.fit_boxsize))

                fitted_star = LinkedEPSFStar(fitted_star)

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        """
        Fit an ePSF model to a single star.

        The input ``epsf`` will usually be modified by the fitting
        routine in this function. Make a copy before calling this
        function if the original is needed.
        """
        if fit_boxsize is not None:
            try:
                xcenter, ycenter = star.cutout_center
                large_slc, _ = overlap_slices(star.shape, fit_boxsize,
                                              (ycenter, xcenter),
                                              mode='strict')
            except (PartialOverlapError, NoOverlapError):
                warnings.warn(f'The star at ({star.center[0]}, '
                              f'{star.center[1]}) cannot be fit because '
                              'its fitting region extends beyond the star '
                              'cutout image.', AstropyUserWarning)

                star = copy.deepcopy(star)
                star._fit_error_status = 1

                return star

            data = star.data[large_slc]
            weights = star.weights[large_slc]

            # define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # use the entire cutout image
            data = star.data
            weights = star.weights

            # define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # fitter doesn't support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = copy.copy(fitter.fit_info)

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # compute the star's fitted position
        x_center = star.cutout_center[0] + fitted_epsf.x_0.value
        y_center = star.cutout_center[1] + fitted_epsf.y_0.value

        if fit_error_status != 2:
            star = copy.deepcopy(star)
            star.cutout_center = (x_center, y_center)
            star.flux = fitted_epsf.flux.value
            star._fit_info = fit_info
            star._fit_error_status = fit_error_status
        else:
            star = copy.deepcopy(star)
            star._fit_error_status = fit_error_status

        return star


class EPSFBuilder:
    """
    Build an effective PSF (ePSF) from stellar cutouts.

    See `Anderson and King (2000; PASP 112, 1360)
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
    and `Anderson (2016; WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_
    for details.

    See `Godden and Blundell (2025; RASTI (in prep))`_ for details on
    additions to ePSF building strategies.

    Parameters
    ----------
    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input ``stars`` along each axis. If ``oversampling`` is a scalar
        then it will be used for both axes. If ``oversampling`` has two
        elements, they must be in ``(y, x)`` order.

    shape : float, tuple of two floats, or `None`, optional
        The output ePSF shape. If `None`, the shape is derived from the
        input stars and ``oversampling``. Even sizes are promoted to the
        next odd value so the ePSF has a well-defined central pixel.

    epsf_class : subclass of `ImagePSF`, optional
        The `ImagePSF` subclass used for constructed ePSF models.

    fitter : `EPSFFitter` object, optional
        The fitter used to refit the ePSF to the stars after each build
        iteration. If `None`, a default `EPSFFitter` is created.

    maxiters : int, optional
        The maximum number of build iterations.

    progress_bar : bool, optional
        Whether to display a progress bar during the iterative build.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel applied to the ePSF after each residual
        update. The predefined ``'quartic'`` and ``'quadratic'`` kernels
        are derived from fourth- and second-degree polynomials,
        respectively. A custom 2D kernel array may also be supplied.

    residual_smoothing_kernel : {'gaussian'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel applied to the stacked residual image
        before it is added back into the ePSF. The predefined
        ``'gaussian'`` option uses a Gaussian kernel matched to the
        oversampled grid.

    recenter_epsf : bool, optional
        Whether to recenter the ePSF after each build iteration. The
        recentering step always uses `centroid_com`, derives its box
        size from the ePSF shape, and runs for at most 10 iterations.

    normalise_epsf : bool, optional
        Whether to normalize the ePSF after each build iteration.

    center_accuracy : float, optional
        The convergence threshold in pixels for fitted star centers. The
        build stops when every successfully fit star moves by less than
        this amount between successive iterations.

    gridpoint_estimation : {'mean', 'median', 'weighted_mean', 'polyfit'}, optional
        The estimator used to combine residual samples in each
        oversampled ePSF grid cell.

    edge_clip : int, optional
        The number of oversampled pixels to zero around the ePSF border
        after each iteration.

    calibrate_ppe : tuple of {'Flux', 'Position'}, optional
        Which PPE calibrations to apply. Include ``'Flux'`` to enable
        in-loop and final flux PPE calibration and ``'Position'`` to
        enable position PPE calibration.

    constrain_stars : tuple of {'Flux', 'Position'}, optional
        Which linked-star constraints to apply after each fitting step.
        Include ``'Flux'`` to constrain linked-star fluxes and
        ``'Position'`` to constrain linked-star positions.

    plot_diagnostics : bool, optional
        Whether to show diagnostic PPE and residual plots.

    residual_star_rms_clip : float or `None`, optional
        Threshold for rejecting whole stars from the residual stack
        based on normalized residual RMS. If `None`, no whole-star
        clipping is applied.

    residual_outlier_clip : float or `None`, optional
        Threshold for rejecting outlying residual samples within each
        oversampled grid cell using a MAD-based clip. If `None`, no
        per-gridpoint clipping is applied.

    residual_min_valid_samples : int, optional
        Minimum number of valid residual samples required to update an
        oversampled grid cell. Cells with fewer samples are left
        unchanged for that iteration.

    residual_despike : bool, optional
        Whether to apply a local despiking step to the stacked residual
        image before smoothing and updating the ePSF.


    Notes
    -----
    If your image contains NaN values, you may see better performance if
    you have the `bottleneck`_ package installed.

    .. _bottleneck:  https://github.com/pydata/bottleneck
    """

    def __init__(self, *, 
                 oversampling=4, 
                 shape=None,
                 epsf_class=ImagePSF, 
                 fitter=EPSFFitter(), 
                 maxiters=10,
                 progress_bar=True, 
                 center_accuracy=1.0e-3,
                 gridpoint_estimation='polyfit',
                 smoothing_kernel='quartic', 
                 residual_smoothing_kernel='gaussian',
                 recenter_epsf=True,
                 normalise_epsf=True,
                 edge_clip=1,
                 calibrate_ppe=('Flux', 'Position'),
                 constrain_stars=('Flux', 'Position'),
                 residual_star_rms_clip=3.0,
                 residual_outlier_clip=3.0,
                 residual_min_valid_samples=5,
                 residual_despike=True,
                 plot_diagnostics=True,):

        if oversampling is None:
            msg = "'oversampling' must be specified"
            raise ValueError(msg)
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        self.normalise_epsf = bool(normalise_epsf)
        if shape is not None:
            self.shape = as_pair('shape', shape, lower_bound=(0, 1))
        else:
            self.shape = shape

        self.recenter_epsf = bool(recenter_epsf)
        self.smoothing_kernel = smoothing_kernel

        if residual_smoothing_kernel == 'gaussian':
            self.residual_smoothing = Gaussian2DKernel(x_stddev=1,
                                                  y_stddev=1,
                                                  x_size=self.oversampling[1],
                                                  y_size=self.oversampling[0])
        elif isinstance(residual_smoothing_kernel, np.ndarray):
            self.residual_smoothing = residual_smoothing_kernel
        elif residual_smoothing_kernel is None:
            self.residual_smoothing = None
        else:
            msg = ("residual_smoothing_kernel must be 'gaussian', a 2D "
                   "numpy array, or None")
            raise ValueError(msg)

        self.residual_smoothing_kernel = residual_smoothing_kernel

        if fitter is None:
            fitter = EPSFFitter()
        if not isinstance(fitter, EPSFFitter):
            msg = 'fitter must be an EPSFFitter instance'
            raise TypeError(msg)
        self.fitter = fitter

        if center_accuracy <= 0.0:
            msg = 'center_accuracy must be a positive number'
            raise ValueError(msg)
        self.center_accuracy_sq = center_accuracy**2

        maxiters = int(maxiters)
        if maxiters <= 0:
            msg = 'maxiters must be a positive number'
            raise ValueError(msg)
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        self.epsf_class = epsf_class
        if isinstance(self.epsf_class, partial):
            candidate = self.epsf_class.func
        else:
            candidate = self.epsf_class
        if not issubclass(candidate, ImagePSF):
            msg = 'epsf_class must be a subclass of ImagePSF'
            raise TypeError(msg)
        
        self.gridpoint_estimation = gridpoint_estimation
        if not self.gridpoint_estimation in ['mean', 'median', 'weighted_mean',
                                             'polyfit']:
            msg = ("gridpoint_estimation must be one of 'mean', 'median', "
                   "'weighted_mean', 'polyfit'")
            raise ValueError(msg)

        self.edge_clip = int(edge_clip)

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

        if residual_star_rms_clip is not None and residual_star_rms_clip <= 0:
            msg = 'residual_star_rms_clip must be positive or None'
            raise ValueError(msg)
        self.residual_star_rms_clip = residual_star_rms_clip

        if residual_outlier_clip is not None and residual_outlier_clip <= 0:
            msg = 'residual_outlier_clip must be positive or None'
            raise ValueError(msg)
        self.residual_outlier_clip = residual_outlier_clip

        residual_min_valid_samples = int(residual_min_valid_samples)
        if residual_min_valid_samples <= 0:
            msg = 'residual_min_valid_samples must be a positive integer'
            raise ValueError(msg)
        self.residual_min_valid_samples = residual_min_valid_samples

        self.residual_despike = bool(residual_despike)

    def __call__(self, stars):
        return self.build_epsf(stars)

    def _create_initial_epsf(self, stars):
        """
        Create an initial `_LegacyEPSFModel` object.

        The initial ePSF data are all zeros.

        If ``shape`` is not specified, the shape of the ePSF data array
        is determined from the shape of the input ``stars`` and the
        oversampling factor. If the size is even along any axis, it will
        be made odd by adding one. The output ePSF will always have odd
        sizes along both axes to ensure a central pixel.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        Returns
        -------
        epsf : `_LegacyEPSFModel`
            The initial ePSF model.
        """
        oversampling = self.oversampling
        shape = self.shape

        # define the ePSF shape
        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 1), check_odd=True)
        else:
            # Stars class should have odd-sized dimensions, and thus we
            # get the oversampled shape as oversampling * len + 1; if
            # len=25, then newlen=101, for example.
            x_shape = (np.ceil(stars._max_shape[1]) * oversampling[1]
                       + 1).astype(int)
            y_shape = (np.ceil(stars._max_shape[0]) * oversampling[0]
                       + 1).astype(int)

            shape = np.array((y_shape, x_shape))

        # verify odd sizes of shape
        shape = [(i + 1) if i % 2 == 0 else i for i in shape]

        data = np.zeros(shape, dtype=float)

        return self.epsf_class(data=data, oversampling=oversampling)

    def _resample_residual(self, star, epsf):
        """
        Compute a normalized residual image in the oversampled ePSF
        grid.

        A normalized residual image is calculated by subtracting the
        normalized ePSF model from the normalized star at the location
        of the star in the undersampled grid. The normalized residual
        image is then resampled from the undersampled star grid to the
        oversampled ePSF grid.

        Parameters
        ----------
        star : `EPSFStar` object
            A single star object.

        epsf : `_LegacyEPSFModel` object
            The ePSF model.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled residual image. The
            image contains NaNs where there is no data.
        """
        # Compute the normalized residual by subtracting the ePSF model
        # from the normalized star at the location of the star in the
        # undersampled grid.

        # Compute the residual image for the star.
        residual_img = star.compute_residual_image(epsf)

        # Normalise the residuals by the star flux
        residual_img /= star.flux

        # Keep only unmasked residual samples so values match the
        # unmasked coordinate vectors (star._xidx_centered/_yidx_centered).
        residual_img = residual_img[~star.mask].ravel()

        # Convert pixel sample positions to the oversampled grid (1D arrays)
        x = epsf.oversampling[1] * star._xidx_centered
        y = epsf.oversampling[0] * star._yidx_centered

        # Compute the location of the ePSF centre in the oversampled grid
        epsf_xcenter, epsf_ycenter = (int((epsf.data.shape[1] - 1) / 2),
                                      int((epsf.data.shape[0] - 1) / 2))
        
        # Convert pixel sample positions to the indexes they belong to in the oversampled grid
        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)

        # Calculate the coordinates of the pixel relative to the index it belongs to in the oversampled grid
        x_coord = x + epsf_xcenter - xidx
        y_coord = y + epsf_ycenter - yidx

        # Calculate the distance between the pixel sample position and the index it belongs to in the oversampled grid. Normalise by the maximum distance a pixel can be from the index it belongs to.
        xdist = np.abs(x_coord) / 0.5
        ydist = np.abs(y_coord) / 0.5

        # Set up empty results arrays (2D arrays with the same shape as the ePSF data array)
        resampled_img = np.full(epsf.data.shape, np.nan)
        img_weights = np.full(epsf.data.shape, 0.0)
        x_coords_img = np.full(epsf.data.shape, np.nan)
        y_coords_img = np.full(epsf.data.shape, np.nan)

        # Mask out any pixel samples that fall outside the bounds of the ePSF data array
        mask = np.logical_and(
            np.logical_and(xidx >= 0, xidx < epsf.data.shape[1]),
            np.logical_and(yidx >= 0, yidx < epsf.data.shape[0]))
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]
        xdist_ = xdist[mask]
        ydist_ = ydist[mask] 
        x_coord_ = x_coord[mask]
        y_coord_ = y_coord[mask]

        # Fill the resampled image with the (masked) residuals
        resampled_img[yidx_, xidx_] = residual_img[mask]

        # Compute the weights for the resampling
        img_weights[yidx_, xidx_] = 1.0 - 1/np.sqrt(2) * np.sqrt(xdist_**2 + ydist_**2)

        # Compute the coordinates of the pixel relative to the index it belongs to in the oversampled grid
        x_coords_img[yidx_, xidx_] = x_coord_
        y_coords_img[yidx_, xidx_] = y_coord_

        return resampled_img, img_weights, x_coords_img, y_coords_img

    def _resample_residuals(self, stars, epsf):
        """
        Compute normalized residual images for all the input stars.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `_LegacyEPSFModel` object
            The ePSF model.

        Returns
        -------
        epsf_resid : 3D `~numpy.ndarray`
            A 3D cube containing the resampled residual images.
        """
        shape = (stars.n_good_stars, epsf.data.shape[0], epsf.data.shape[1])
        epsf_resid = np.zeros(shape)
        epsf_resid_weights = np.zeros(shape)
        epsf_x_coords = np.zeros(shape)
        epsf_y_coords = np.zeros(shape)
        for i, star in enumerate(stars.all_good_stars):
            resampled_img, img_weights, x_coords_img, y_coords_img = self._resample_residual(star, epsf)
            epsf_resid[i, :, :] = resampled_img
            epsf_resid_weights[i, :, :] = img_weights
            epsf_x_coords[i, :, :] = x_coords_img
            epsf_y_coords[i, :, :] = y_coords_img
        return epsf_resid, epsf_resid_weights, epsf_x_coords, epsf_y_coords

    def _smooth_epsf(self, epsf_data):
        """
        Smooth the ePSF array by convolving it with a kernel.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            A 2D array containing the ePSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The smoothed (convolved) ePSF data.
        """
        if self.smoothing_kernel is None:
            return epsf_data

        # do this check first as comparing a ndarray to string causes a warning
        if isinstance(self.smoothing_kernel, np.ndarray):
            kernel = self.smoothing_kernel

        elif self.smoothing_kernel == 'quartic':
            # from Polynomial2D fit with degree=4 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(4, c0_0=0.04163265, c1_0=-0.76326531,
            #              c2_0=0.99081633, c3_0=-0.4, c4_0=0.05,
            #              c0_1=-0.76326531, c0_2=0.99081633, c0_3=-0.4,
            #              c0_4=0.05, c1_1=0.32653061, c1_2=-0.08163265,
            #              c1_3=0.0, c2_1=-0.08163265, c2_2=0.02040816,
            #              c3_1=-0.0)>
            kernel = np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

        elif self.smoothing_kernel == 'quadratic':
            # from Polynomial2D fit with degree=2 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(2, c0_0=-0.07428571, c1_0=0.11428571,
            #              c2_0=-0.02857143, c0_1=0.11428571,
            #              c0_2=-0.02857143, c1_1=-0.0)
            kernel = np.array(
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

        else:
            msg = 'Unsupported kernel'
            raise TypeError(msg)

        return convolve(epsf_data, kernel)

    def _recenter_epsf(self, epsf, center_accuracy=1.0e-4):
        """
        Calculate the center of the ePSF data and shift the data so the
        ePSF center is at the center of the ePSF data array.

        Parameters
        ----------
        epsf : `_LegacyEPSFModel` object
            The ePSF model.

        center_accuracy : float, optional
            The desired accuracy for the centers of stars. The building
            iterations will stop if the center of the ePSF changes by
            less than ``center_accuracy`` pixels between iterations.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data.
        """
        epsf_data = epsf.data

        epsf = self.epsf_class(data=epsf.data,
                                oversampling=epsf.oversampling,
                                origin=epsf.origin)
        maxiters = 10
        box_size = np.rint((np.asarray(epsf.data.shape, dtype=float) - 1.0)
                           / np.asarray(epsf.oversampling,
                                        dtype=float)).astype(int)
        box_size = np.maximum(box_size, 3)
        box_size = np.where(box_size % 2 == 0, box_size + 1, box_size)

        xcenter, ycenter = epsf.origin

        y, x = np.indices(epsf.data.shape, dtype=float)
        x /= epsf.oversampling[1]
        y /= epsf.oversampling[0]

        dx_total, dy_total = 0, 0
        iter_num = 0
        center_accuracy_sq = center_accuracy**2
        center_dist_sq = center_accuracy_sq + 1.0e6
        center_dist_sq_prev = center_dist_sq + 1
        while (iter_num < maxiters and center_dist_sq >= center_accuracy_sq):
            iter_num += 1

            # Anderson & King (2000) recentering function depends
            # on specific pixels, and thus does not need a cutout
            slices_large, _ = overlap_slices(epsf_data.shape, 
                                             box_size * self.oversampling,
                                             (ycenter, xcenter))
            epsf_cutout = epsf_data[slices_large]
            mask = ~np.isfinite(epsf_cutout)

            # find a new center position
            xcenter_new, ycenter_new = centroid_com(epsf_cutout,
                                                    mask=mask)

            xcenter_new += slices_large[1].start
            ycenter_new += slices_large[0].start

            # Calculate the shift; dx = i - x_star so if dx was positively
            # incremented then x_star was negatively incremented for a given i.
            # We will therefore actually subsequently subtract dx from xcenter
            # (or x_star).
            dx = (xcenter_new - xcenter) / epsf.oversampling[1]
            dy = (ycenter_new - ycenter) / epsf.oversampling[0]

            center_dist_sq = dx**2 + dy**2

            if center_dist_sq >= center_dist_sq_prev:  # don't shift
                break
            center_dist_sq_prev = center_dist_sq

            dx_total += dx
            dy_total += dy

            new_x_0 = (xcenter / epsf.oversampling[1]) - dx_total
            new_y_0 = (ycenter / epsf.oversampling[0]) - dy_total

            epsf_data = epsf.evaluate(x=x, y=y, flux=1.0,
                                      x_0=new_x_0,
                                      y_0=new_y_0)

        return epsf_data
    
    def _normalise_epsf(self, epsf):
        """
        Normalize the ePSF data.

        Parameters
        ----------
        epsf: ImagePSF object
            The ePSF model.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The normalized ePSF data.
        """

        # Convert box size to integer
        box_size = np.rint((np.asarray(epsf.data.shape, dtype=float) - 1.0)
                           / np.asarray(self.oversampling,
                                        dtype=float)).astype(int)
        box_size = as_pair('box_size', box_size,
                            lower_bound=(3, 3), check_odd=False)
            
        # Check that box size is >= 3
        if box_size[0] < 3 or box_size[1] < 3:
            msg = 'box_size values must be >= 3'
            raise ValueError(msg)
            
        # Get the pixel positions at which to normalize the ePSF
        half_x = int(np.floor(box_size[1]/2))
        half_y = int(np.floor(box_size[0]/2))
        norm_x = np.arange(-half_x, half_x + 1)
        norm_y = np.arange(-half_y, half_y + 1)

        yy, xx = np.meshgrid(norm_y, norm_x)

        # Evaluate the ePSF at these pixel positions
        epsf_values = epsf.evaluate(x=xx, y=yy, flux=1.0, x_0=0.0, y_0=0.0)

        # Normalize the ePSF data
        total = np.nansum(epsf_values)
        if total > 0.0:
            epsf_data = epsf.data / total
        else:
            # Warning: if the total is zero or negative, we cannot normalize, so we return the original data and issue a warning.
            warnings.warn('Cannot normalize ePSF because total is non-positive. Returning original ePSF data.', AstropyUserWarning)
            epsf_data = epsf.data

        return epsf_data

    @staticmethod
    def _mad_std(values):
        """
        Return a robust standard-deviation estimate from the MAD.
        """
        values = np.asanyarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        return 1.4826 * mad

    def _select_residual_stars(self, stars, epsf):
        """
        Select stars for the residual stack using a robust RMS clip.
        """
        good_stars = stars.all_good_stars
        if len(good_stars) == 0 or self.residual_star_rms_clip is None:
            return good_stars

        rms_values = []
        for star in good_stars:
            flux = float(np.asanyarray(star.flux, dtype=float))
            if not np.isfinite(flux) or flux == 0.0:
                rms_values.append(np.nan)
                continue

            residual = star.compute_residual_image(epsf) / flux
            residual = residual[~star.mask]
            residual = residual[np.isfinite(residual)]
            if residual.size == 0:
                rms_values.append(np.nan)
            else:
                rms_values.append(np.sqrt(np.nanmean(residual**2)))

        rms_values = np.asarray(rms_values, dtype=float)
        valid = np.isfinite(rms_values)
        if np.count_nonzero(valid) < 3:
            return good_stars

        center = np.nanmedian(rms_values[valid])
        scale = self._mad_std(rms_values[valid])
        if not np.isfinite(scale) or scale == 0.0:
            return good_stars

        threshold = center + self.residual_star_rms_clip * scale
        selected = [star for star, rms in zip(good_stars, rms_values,
                                              strict=True)
                    if np.isfinite(rms) and rms <= threshold]

        return selected if len(selected) > 0 else good_stars

    def _combine_residual_stack(self, residuals, weights, x_coords, y_coords):
        """
        Combine a residual stack into a single robust 2D residual image.
        """
        _, ny, nx = residuals.shape
        combined = np.full((ny, nx), np.nan)
        counts = np.zeros((ny, nx), dtype=int)

        for i in range(ny):
            for j in range(nx):
                z = residuals[:, i, j]
                valid = np.isfinite(z)
                valid_count = np.count_nonzero(valid)
                counts[i, j] = valid_count
                if valid_count < self.residual_min_valid_samples:
                    continue

                z = z[valid]
                w = weights[valid, i, j]
                x = x_coords[valid, i, j]
                y = y_coords[valid, i, j]

                if self.residual_outlier_clip is not None and z.size >= 3:
                    center = np.nanmedian(z)
                    scale = self._mad_std(z)
                    if np.isfinite(scale) and scale > 0.0:
                        keep = np.abs(z - center) <= (self.residual_outlier_clip
                                                      * scale)
                        if np.count_nonzero(keep) >= self.residual_min_valid_samples:
                            z = z[keep]
                            w = w[keep]
                            x = x[keep]
                            y = y[keep]

                if z.size < self.residual_min_valid_samples:
                    continue

                if self.gridpoint_estimation == 'mean':
                    combined[i, j] = np.nanmean(z)
                elif self.gridpoint_estimation == 'median':
                    combined[i, j] = np.nanmedian(z)
                elif self.gridpoint_estimation == 'weighted_mean':
                    valid_w = np.isfinite(w) & (w > 0.0)
                    if np.count_nonzero(valid_w) < self.residual_min_valid_samples:
                        continue
                    denom = np.nansum(w[valid_w])
                    if denom > 0.0:
                        combined[i, j] = np.nansum(z[valid_w] * w[valid_w]) / denom
                elif self.gridpoint_estimation == 'polyfit':
                    if z.size <= 10:
                        combined[i, j] = np.nanmedian(z)
                    else:
                        A = np.column_stack((np.ones_like(x), x, y,
                                             x**2, x * y, y**2))
                        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
                        combined[i, j] = coeffs[0]
                else:
                    msg = 'Unsupported gridpoint_estimation method'
                    raise TypeError(msg)

        return combined, counts

    def _despike_residuals(self, residuals):
        """
        Replace isolated hot residual cells with the local median.
        """
        if not self.residual_despike:
            return residuals

        residual_despike_boxsize = (3, 3)
        residual_despike_passes = 2
        residual_despike_threshold = 3.0

        footprint = np.ones(residual_despike_boxsize, dtype=bool)
        center = tuple(size // 2 for size in residual_despike_boxsize)
        footprint[center] = False

        despiked = residuals.copy()
        hot_mask = np.zeros_like(residuals, dtype=bool)

        for _ in range(residual_despike_passes):
            local_median = median_filter(despiked, footprint=footprint,
                                         mode='nearest')

            local_abs_dev = median_filter(np.abs(despiked - local_median),
                                          footprint=footprint,
                                          mode='nearest')
            local_scale = 1.4826 * local_abs_dev

            hot_mask = np.abs(despiked - local_median) > (
                residual_despike_threshold * local_scale)
            hot_mask &= np.isfinite(despiked)
            hot_mask &= np.isfinite(local_median)

            if not np.any(hot_mask):
                break

            despiked[hot_mask] = local_median[hot_mask]

        # TEMP: Plot the hot mask for diagnostics
        if self.plot_diagnostics:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.imshow(hot_mask, origin='lower', cmap='Reds')
            plt.title('Hot Residual Mask')
            plt.xlabel('X Pixel Index')
            plt.ylabel('Y Pixel Index')
            plt.colorbar(label='Hot Mask Value')
            plt.show()

        return despiked

    def _plot_residual_quadrant_stacks(self, stack_stars, residuals, weights,
                                       x_coords, y_coords):
        """
        Plot combined residual stacks split by detector-position quadrant.
        """
        if not self.plot_diagnostics or len(stack_stars) == 0:
            return

        centers = np.asarray([star.center for star in stack_stars], dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            return

        xmin, ymin = np.min(centers, axis=0)
        xmax, ymax = np.max(centers, axis=0)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        quadrant_masks = (
            ((centers[:, 0] <= xmid) & (centers[:, 1] <= ymid), 'Lower Left'),
            ((centers[:, 0] > xmid) & (centers[:, 1] <= ymid), 'Lower Right'),
            ((centers[:, 0] <= xmid) & (centers[:, 1] > ymid), 'Upper Left'),
            ((centers[:, 0] > xmid) & (centers[:, 1] > ymid), 'Upper Right'),
        )

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                 constrained_layout=True)
        axes = axes.ravel()

        for ax, (mask, title) in zip(axes, quadrant_masks, strict=True):
            if not np.any(mask):
                combined = np.full(residuals.shape[1:], np.nan)
            else:
                combined, _ = self._combine_residual_stack(
                    residuals[mask], weights[mask], x_coords[mask], y_coords[mask])

            im = ax.imshow(combined, origin='lower', cmap='coolwarm')
            ax.set_title(f'{title} ({np.count_nonzero(mask)} stars)')
            ax.set_xlabel('X Pixel Index')
            ax.set_ylabel('Y Pixel Index')
            fig.colorbar(im, ax=ax, label='Residual Value')

        fig.suptitle('Combined Residual Stacks by Image Region')
        plt.show()

    def _plot_residual_flux_stacks(self, stack_stars, residuals, weights,
                                   x_coords, y_coords):
        """
        Plot combined residual stacks split by stellar flux.
        """
        if not self.plot_diagnostics or len(stack_stars) == 0:
            return

        fluxes = np.asarray([star.flux for star in stack_stars], dtype=float)
        valid = np.isfinite(fluxes)
        if np.count_nonzero(valid) == 0:
            return

        fluxes_valid = fluxes[valid]
        fmin = np.min(fluxes_valid)
        fmax = np.max(fluxes_valid)
        if fmin == fmax:
            bin_masks = (
                (valid, 'All Fluxes'),
                (np.zeros_like(valid, dtype=bool), 'Mid Flux'),
                (np.zeros_like(valid, dtype=bool), 'High Flux'),
            )
        else:
            edges = np.linspace(fmin, fmax, 4)
            bin_masks = (
                (valid & (fluxes >= edges[0]) & (fluxes <= edges[1]),
                 f'Low Flux [{edges[0]:.3g}, {edges[1]:.3g}]'),
                (valid & (fluxes > edges[1]) & (fluxes <= edges[2]),
                 f'Mid Flux ({edges[1]:.3g}, {edges[2]:.3g}]'),
                (valid & (fluxes > edges[2]) & (fluxes <= edges[3]),
                 f'High Flux ({edges[2]:.3g}, {edges[3]:.3g}]'),
            )

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                 constrained_layout=True)

        for ax, (mask, title) in zip(axes, bin_masks, strict=True):
            if not np.any(mask):
                combined = np.full(residuals.shape[1:], np.nan)
            else:
                combined, _ = self._combine_residual_stack(
                    residuals[mask], weights[mask], x_coords[mask], y_coords[mask])

            im = ax.imshow(combined, origin='lower', cmap='coolwarm')
            ax.set_title(f'{title}\n({np.count_nonzero(mask)} stars)')
            ax.set_xlabel('X Pixel Index')
            ax.set_ylabel('Y Pixel Index')
            fig.colorbar(im, ax=ax, label='Residual Value')

        fig.suptitle('Combined Residual Stacks by Flux Bin')
        plt.show()

    def _build_epsf_step(self, stars, epsf=None, *, iter_num=None):
        """
        A single iteration of improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `_LegacyEPSFModel` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `_LegacyEPSFModel` object
            The updated ePSF.
        """
        if len(stars) < 1:
            msg = ('stars must contain at least one EPSFStar or '
                   'LinkedEPSFStar object')
            raise ValueError(msg)

        if epsf is None:
            # create an initial ePSF (array of zeros)
            epsf = self._create_initial_epsf(stars)
        else:
            # improve the input ePSF
            epsf = copy.deepcopy(epsf)

        stack_stars = self._select_residual_stars(stars, epsf)
        if len(stack_stars) == 0:
            stack_stars = stars.all_good_stars

        # compute a 3D stack of 2D residual images
        residuals, weights, x_coords, y_coords = self._resample_residuals(
            EPSFStars(stack_stars), epsf)
        
        # TODO: Plot diagnostic plot for the residuals

        residuals, residual_counts = self._combine_residual_stack(
            residuals, weights, x_coords, y_coords)
        
        if self.plot_diagnostics:
            self._plot_residual_image(residuals)

        # Leave underconstrained cells unchanged in this iteration.
        residuals[~np.isfinite(residuals)] = 0.0
        if iter_num is None or iter_num >= 2:
            residuals = self._despike_residuals(residuals)

        if self.residual_smoothing is not None:
            # Smooth the residuals
            residuals = convolve(residuals, self.residual_smoothing)

        if self.plot_diagnostics:
            self._plot_residual_image(residuals)

        # add the residuals to the previous ePSF image
        new_epsf_data = epsf.data + (0.8 * residuals)

        # smooth and recenter the ePSF
        smoothed_data = self._smooth_epsf(new_epsf_data)

        epsf = self.epsf_class(data=smoothed_data,
                                oversampling=epsf.oversampling,
                                origin=epsf.origin)

        if self.recenter_epsf:
            recentered_data = self._recenter_epsf(epsf)
        else:
            recentered_data = smoothed_data

        epsf = self.epsf_class(data=recentered_data,
                                oversampling=epsf.oversampling,
                                origin=epsf.origin)
        
        # Normalize the ePSF
        if self.normalise_epsf:
            normalised_data = self._normalise_epsf(epsf)
        else:
            normalised_data = recentered_data

        # Clip the edges of the ePSF data if required
        if self.edge_clip and self.edge_clip > 0:
            clipped_data = normalised_data.copy()
            clipped_data[:self.edge_clip, :] = 0.0
            clipped_data[-self.edge_clip:, :] = 0.0
            clipped_data[:, :self.edge_clip] = 0.0
            clipped_data[:, -self.edge_clip:] = 0.0
        else:
            clipped_data = normalised_data

        return_epsf = self.epsf_class(data=clipped_data,
                                oversampling=epsf.oversampling,
                                origin=epsf.origin)
        
        if self.plot_diagnostics:
            self._plot_epsf(return_epsf)

        return return_epsf
    
    def _resample_epsf(self, epsf, oversampling):
        """
        Resample the input ePSF to the oversampling factor used in the build process.

        Parameters
        ----------
        epsf : `ImagePSF` object
            The input ePSF model.

        oversampling : tuple of two ints
            The oversampling factor used in the build process.

        Returns
        -------
        resampled_epsf_data : 2D `~numpy.ndarray`
            A 2D array containing the resampled ePSF data.
        """

        # Convert from oversampled-array dimensions to the equivalent
        # image-grid dimensions using grid endpoints, then convert to
        # the requested oversampling.
        input_shape = ((np.asarray(epsf.data.shape, dtype=float) - 1.0)
                       / np.asarray(epsf.oversampling, dtype=float))
        output_shape = np.rint(input_shape * np.asarray(oversampling)
                               + 1.0).astype(int)

        # Ensure odd output dimensions so the central pixel is well defined.
        output_shape = np.where(output_shape % 2 == 0, output_shape + 1,
                                output_shape)

        output_center_yx = (output_shape - 1.0) / 2.0

        # Evaluate on the requested image-coordinate grid.
        y = (np.arange(output_shape[0], dtype=float) - output_center_yx[0])
        y /= oversampling[0]
        x = (np.arange(output_shape[1], dtype=float) - output_center_yx[1])
        x /= oversampling[1]
        yy, xx = np.meshgrid(y, x, indexing='ij')

        # Evaluate the input ePSF at these x and y values to get the resampled ePSF data
        resampled_epsf_data = epsf.evaluate(x=xx, y=yy, flux=1.0, x_0=0.0, y_0=0.0)

        return resampled_epsf_data
    
    def _collect_ppe_residual_results(self, stars):
        """
        Collect PPE residual samples from linked stars.
        """
        residual_results = {'subpixel_x': [], 'subpixel_y': [],
                            'x_residual': [], 'y_residual': [],
                            'flux_residual': [], 'center_x': [],
                            'center_y': []}

        # Iterate over the linked stars in stars
        for linked_star in stars._data:
            if not isinstance(linked_star, LinkedEPSFStar):
                continue
            # Check that linked star has at least 3 good stars to ensure a reliable mean position and flux.
            if len(linked_star.all_good_stars) < 3:
                continue
            mean_flux = linked_star.get_mean_flux()
            mean_ra, mean_dec = linked_star.get_mean_radec()
            if mean_flux is None or not np.isfinite(mean_flux):
                continue

            # Iterate over each star in the linked star.
            for star in linked_star.all_good_stars:
                if star.wcs_large is None:
                    continue

                # Calculate the linked-star mean sky position projected
                # onto this detector frame.
                mean_x, mean_y = star.wcs_large.world_to_pixel_values(
                    mean_ra, mean_dec)
                if not (np.isfinite(mean_x) and np.isfinite(mean_y)):
                    continue

                # Calculate the measured-minus-mean residuals.
                x_residual = star.center[0] - mean_x
                y_residual = star.center[1] - mean_y

                if mean_flux == 0.0:
                    flux_residual = 0.0
                else:
                    flux_residual = (star.flux - mean_flux) / mean_flux

                # Record the residuals with the measured (uncorrected)
                # subpixel position of the star.
                residual_results['subpixel_x'].append(np.mod(star.center[0],
                                                             1.0))
                residual_results['subpixel_y'].append(np.mod(star.center[1],
                                                             1.0))
                residual_results['x_residual'].append(x_residual)
                residual_results['y_residual'].append(y_residual)
                residual_results['flux_residual'].append(flux_residual)
                residual_results['center_x'].append(star.center[0])
                residual_results['center_y'].append(star.center[1])

        return residual_results

    def _generate_ppe_map_from_results(self, residual_results,
                                       supersampling=None):
        """
        Generate a PPE map from precomputed residual samples.
        """
        if supersampling is None:
            supersampling = self.oversampling
        supersampling = as_pair('supersampling', supersampling,
                                lower_bound=(1, 1))

        # Set up results arrays for the PPE maps
        ppe_flux_map = np.full((supersampling[0], supersampling[1]), np.nan)
        ppe_x_map = np.full((supersampling[0], supersampling[1]), np.nan)
        ppe_y_map = np.full((supersampling[0], supersampling[1]), np.nan)

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

        # Generate the PPE map arrays from the residual results using
        # the median value in each supersampled subpixel bin.
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

        # Generate a PPEMap object from the PPE map arrays.
        return PPEMap(supersampling, ppe_flux_map, ppe_x_map, ppe_y_map)

    def _generate_ppe_map(self, stars, supersampling=None):
        """
        Generate supersampled 2D PPE maps for the flux and position
        measurements of the stars.
        """
        residual_results = self._collect_ppe_residual_results(stars)
        return self._generate_ppe_map_from_results(residual_results,
                                                   supersampling=supersampling)

    def _plot_ppe_region_diagnostics(self, stars, supersampling=None):
        """
        Plot PPE maps split by detector-position quadrant.
        """
        if not self.plot_diagnostics:
            return None

        residual_results = self._collect_ppe_residual_results(stars)
        if len(residual_results['subpixel_x']) == 0:
            return None

        center_x = np.asarray(residual_results['center_x'], dtype=float)
        center_y = np.asarray(residual_results['center_y'], dtype=float)
        xmin = np.min(center_x)
        xmax = np.max(center_x)
        ymin = np.min(center_y)
        ymax = np.max(center_y)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        quadrant_masks = (
            ((center_x <= xmid) & (center_y <= ymid), 'Lower Left'),
            ((center_x > xmid) & (center_y <= ymid), 'Lower Right'),
            ((center_x <= xmid) & (center_y > ymid), 'Upper Left'),
            ((center_x > xmid) & (center_y > ymid), 'Upper Right'),
        )

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 3, figsize=(12, 14),
                                 constrained_layout=True)
        sample_factor = 3
        map_specs = (
            ('flux_ppe', 'Flux PPE', 'Fractional flux residual'),
            ('x_ppe', 'X PPE', 'X residual (pixels)'),
            ('y_ppe', 'Y PPE', 'Y residual (pixels)'),
        )
        extent = (0.0, 1.0, 0.0, 1.0)

        for row, (mask, region_title) in enumerate(quadrant_masks):
            region_results = {}
            for key, values in residual_results.items():
                arr = np.asarray(values)
                region_results[key] = arr[mask]

            region_map = self._generate_ppe_map_from_results(
                region_results, supersampling=supersampling)

            yphase = np.linspace(0.0, 1.0,
                                 region_map.supersampling[0] * sample_factor,
                                 endpoint=False)
            xphase = np.linspace(0.0, 1.0,
                                 region_map.supersampling[1] * sample_factor,
                                 endpoint=False)
            xx, yy = np.meshgrid(xphase, yphase)

            sampled_maps = (
                region_map._sample_periodic_map(region_map.flux_ppe, xx, yy,
                                                region_map.supersampling),
                region_map._sample_periodic_map(region_map.x_ppe, xx, yy,
                                                region_map.supersampling),
                region_map._sample_periodic_map(region_map.y_ppe, xx, yy,
                                                region_map.supersampling),
            )

            for col, ((_, title, cbar_label), data) in enumerate(
                    zip(map_specs, sampled_maps, strict=True)):
                ax = axes[row, col]
                im = ax.imshow(data, origin='lower', extent=extent,
                               cmap='coolwarm', aspect='equal')
                ax.set_title(f'{region_title}: {title}')
                ax.set_xlabel('Subpixel x phase')
                ax.set_ylabel('Subpixel y phase')
                fig.colorbar(im, ax=ax, label=cbar_label)

        return fig

    def _plot_diagnostics(self, stars, ppe_map):
        """
        Plot PPE diagnostics when enabled.
        """
        if not self.plot_diagnostics:
            return None

        ppe_map.plot_maps()
        return self._plot_ppe_region_diagnostics(stars)
    
    def _plot_star_distribution(self, stars):
        """
        Plot the distribution of the star sample in image coordinates and sub-pixel coordinates, and the coverage of the sub-pixel bins by the star sample.
        """
        if not self.plot_diagnostics:
            return None

        # Plot 1: Star sample distributions
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # Subfigure 1: Image positions of the stars color-coded by flux
        centers = np.asarray([star.center for star in stars.all_good_stars], dtype=float)
        fluxes = np.asarray([star.flux for star in stars.all_good_stars], dtype=float)
        sc = axes[0].scatter(centers[:, 0], centers[:, 1], c=fluxes, s=10, alpha=0.7, cmap='viridis')
        axes[0].set_title('Star Image Positions')
        axes[0].set_xlabel('X Pixel Coordinate')
        axes[0].set_ylabel('Y Pixel Coordinate')
        axes[0].set_aspect('equal')
        cbar = fig.colorbar(sc, ax=axes[0])
        cbar.set_label('Flux (counts per second)')
        # Subfigure 2: Sub-pixel positions of the stars
        # Wrap to [-0.5, 0.5) so all points map into finite-width bins
        # without requiring an extra empty edge bin.
        subpixel_x = ((np.mod(centers[:, 0], 1.0) + 0.5) % 1.0) - 0.5
        subpixel_y = ((np.mod(centers[:, 1], 1.0) + 0.5) % 1.0) - 0.5
        axes[1].scatter(subpixel_x, subpixel_y, c=fluxes, s=10, alpha=0.7, cmap='viridis')
        # Set the limits to show the full range of sub-pixel positions    
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_ylim(-0.5, 0.5)

        n_bins_x = int(self.oversampling[1])
        n_bins_y = int(self.oversampling[0])
        x_edges = np.linspace(-0.5, 0.5, n_bins_x + 1)
        y_edges = np.linspace(-0.5, 0.5, n_bins_y + 1)

        # Show subpixel-gridsection boundaries for better visualization
        for b in x_edges[1:-1]:
            axes[1].axvline(b, color='gray', linestyle='--', linewidth=0.5)
        for b in y_edges[1:-1]:
            axes[1].axhline(b, color='gray', linestyle='--', linewidth=0.5)
        # Set the title and labels for the sub-pixel position plot
        axes[1].set_title('Star Sub-Pixel Positions')
        axes[1].set_xlabel('Sub-Pixel X Phase')
        axes[1].set_ylabel('Sub-Pixel Y Phase')
        axes[1].set_aspect('equal')
        cbar = fig.colorbar(sc, ax=axes[1])
        cbar.set_label('Flux (counts per second)')
        # Subfigure 3: Count the number of stars in each sub-pixel bin
        xbin = np.searchsorted(x_edges, subpixel_x, side='right') - 1
        ybin = np.searchsorted(y_edges, subpixel_y, side='right') - 1
        xbin = np.clip(xbin, 0, n_bins_x - 1)
        ybin = np.clip(ybin, 0, n_bins_y - 1)

        bin_counts = np.zeros((n_bins_y, n_bins_x), dtype=int)
        np.add.at(bin_counts, (ybin, xbin), 1)

        im = axes[2].imshow(bin_counts, origin='lower', cmap='plasma')
        axes[2].set_title('Star Counts in Sub-Pixel Bins')
        axes[2].set_xlabel('Sub-Pixel X Bin')
        axes[2].set_ylabel('Sub-Pixel Y Bin')
        cbar = fig.colorbar(im, ax=axes[2])
        cbar.set_label('Number of Stars')
        # Subfigure 4: Number of unique star id_labels in each sub-pixel bin (to check for linked-star coverage)
        id_labels = np.asarray([star.id_label for star in stars.all_good_stars])
        id_bin_counts = np.zeros((n_bins_y, n_bins_x), dtype=int)
        id_sets = [[set() for _ in range(n_bins_x)] for _ in range(n_bins_y)]
        for xb, yb, id_label in zip(xbin, ybin, id_labels, strict=True):
            id_sets[yb][xb].add(id_label)
        for j in range(n_bins_y):
            for i in range(n_bins_x):
                id_bin_counts[j, i] = len(id_sets[j][i])

        im = axes[3].imshow(id_bin_counts, origin='lower', cmap='inferno')
        axes[3].set_title('Unique Star ID Counts in Sub-Pixel Bins')
        axes[3].set_xlabel('Sub-Pixel X Bin')
        axes[3].set_ylabel('Sub-Pixel Y Bin')
        cbar = fig.colorbar(im, ax=axes[3])
        cbar.set_label('Number of Unique Star IDs')
        plt.show()

    def _plot_residual_image(self, residuals):
        """
        Plot the combined residual image.
        """
        if not self.plot_diagnostics:
            return None

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(residuals, origin='lower', cmap='coolwarm')
        plt.title('Combined Residual Image')
        plt.xlabel('X Pixel Index')
        plt.ylabel('Y Pixel Index')
        plt.colorbar(label='Residual Value')
        plt.show()

    def _plot_epsf(self, epsf):
        """
        Plot the ePSF image.
        """
        if not self.plot_diagnostics:
            return None

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(epsf.data, origin='lower', cmap='viridis')
        plt.title('ePSF Image')
        plt.xlabel('X Pixel Index')
        plt.ylabel('Y Pixel Index')
        plt.colorbar(label='ePSF Value')
        plt.show()

    def _correct_stars_ppe(self, stars, ppe_map, *, apply_flux=True,
                           apply_position=True, flux_damping=1.0):
        """
        Apply PPE corrections to the stars' fitted fluxes and centers.
        """
        corrected_stars = copy.deepcopy(stars)

        for star in corrected_stars.all_stars:
            corrected_flux, corrected_x, corrected_y = ppe_map(
                star.flux, star.center[0], star.center[1])
            if apply_flux:
                star.flux = (star.flux
                             + flux_damping * (corrected_flux - star.flux))
            if apply_position:
                star.cutout_center = np.array((corrected_x, corrected_y),
                                              dtype=float) - star.origin

        return corrected_stars

    def _apply_ppe_corrections(self, stars, ppe_map, *, iteration=None,
                               final=False):
        """
        Apply PPE corrections using the configured in-loop/final policy.
        """
        apply_position = 'Position' in self.calibrate_ppe
        apply_flux = False
        flux_damping = 0.8

        if final:
            apply_flux = 'Flux' in self.calibrate_ppe
            flux_damping = 1.0
        elif 'Flux' in self.calibrate_ppe and iteration is not None:
            apply_flux = True

        if not apply_position and not apply_flux:
            return stars

        return self._correct_stars_ppe(stars, ppe_map,
                                       apply_flux=apply_flux,
                                       apply_position=apply_position,
                                       flux_damping=flux_damping)

    def _apply_linked_star_constraints(self, stars):
        """
        Apply the configured linked-star constraints.
        """
        if 'Position' in self.constrain_stars:
            stars.constrain_linked_centres()
        if 'Flux' in self.constrain_stars:
            stars.constrain_linked_fluxes()
        return stars

    def _get_init_epsf(self, epsf):
        """
        Obtain the initial ePSF model from the input ePSF model, noting that the input ePSF model may not have the same oversampling factor as the ePSF model used in the build process. The initial ePSF should be constructed by resampling the input ePSF to the oversampling factor and origin used in the build process."""
        
        if not isinstance(epsf, ImagePSF):
            return None
        
        if np.array_equal(np.asarray(epsf.oversampling),
                          np.asarray(self.oversampling)):
            return epsf
        
        resampled_epsf_data = self._resample_epsf(epsf, self.oversampling)
        
        return self.epsf_class(data=resampled_epsf_data, oversampling=self.oversampling)

    def build_epsf(self, stars, *, init_model=None):
        """
        Build iteratively an ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        init_model : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `ImagePSF` object
            The constructed ePSF.

        fitted_stars : `EPSFStars` object
            The input stars with updated centers and fluxes derived
            from fitting the output ``epsf``.
        """
        iter_num = 0
        fit_failed = np.zeros(stars.n_stars, dtype=bool)
        epsf = self._get_init_epsf(init_model)
        center_dist_sq = self.center_accuracy_sq + 1.0
        centers = stars.cutout_center_flat

        pbar = None
        if self.progress_bar:
            desc = f'EPSFBuilder ({self.maxiters} maxiters)'
            pbar = add_progress_bar(total=self.maxiters,
                                    desc=desc)  # pragma: no cover

        if epsf is None:
            legacy_epsf = None
        else:
            legacy_epsf = self.epsf_class(epsf.data, flux=epsf.flux,
                                           x_0=epsf.x_0, y_0=epsf.y_0, origin=epsf.origin,
                                           oversampling=epsf.oversampling)
            
        # Initial fit of the ePSF to the stars
        if legacy_epsf is not None:
            with warnings.catch_warnings():
                message = '.*The fit may be unsuccessful;.*'
                warnings.filterwarnings('ignore', message=message,
                                        category=AstropyUserWarning)

                image_psf = self.epsf_class(data=legacy_epsf.data,
                                     origin=legacy_epsf.origin,
                                     oversampling=legacy_epsf.oversampling,
                                     fill_value=legacy_epsf.fill_value)

                stars = self.fitter(image_psf, stars)

        ppe_map = self._generate_ppe_map(stars)

        if self.plot_diagnostics:
            self._plot_star_distribution(stars)
            ppe_map.plot_maps()

        if self.calibrate_ppe:
            stars = self._apply_ppe_corrections(stars, ppe_map, iteration=0)

        if self.constrain_stars:
            stars = self._apply_linked_star_constraints(stars)

        converged = False
        while iter_num < self.maxiters and not np.all(fit_failed):

            iter_num += 1

            if iter_num == 1 and self.residual_smoothing_kernel is not None and legacy_epsf is None:
                # Do not use residual smoothing in the first iteration UNLESS 
                # an initial ePSF is provided
                residual_smoothing_backup = self.residual_smoothing
                self.residual_smoothing = None
            elif iter_num == 2 and self.residual_smoothing_kernel is not None and legacy_epsf is None:
                # Restore residual smoothing after the first iteration if it was disabled
                self.residual_smoothing = residual_smoothing_backup


            if self.calibrate_ppe:
                stars = self._apply_ppe_corrections(stars, ppe_map,
                                                iteration=iter_num)
            if self.constrain_stars:
                stars = self._apply_linked_star_constraints(stars)


            # build/improve the ePSF
            legacy_epsf = self._build_epsf_step(stars, epsf=legacy_epsf,
                                                iter_num=iter_num)

            # fit the new ePSF to the stars to find improved centers
            # we catch fit warnings here -- stars with unsuccessful fits
            # are excluded from the ePSF build process
            with warnings.catch_warnings():
                message = '.*The fit may be unsuccessful;.*'
                warnings.filterwarnings('ignore', message=message,
                                        category=AstropyUserWarning)

                image_psf = self.epsf_class(data=legacy_epsf.data,
                                     origin=legacy_epsf.origin,
                                     oversampling=legacy_epsf.oversampling,
                                     fill_value=legacy_epsf.fill_value)

                stars = self.fitter(image_psf, stars)

            ppe_map = self._generate_ppe_map(stars)

            if self.plot_diagnostics:
                ppe_map.plot_maps()

            # find all stars where the fit failed
            fit_failed = np.array([star._fit_error_status > 0
                                   for star in stars.all_stars])
            if np.all(fit_failed):
                msg = 'The ePSF fitting failed for all stars.'
                raise ValueError(msg)

            # permanently exclude fitting any star where the fit fails
            # after 3 iterations
            if iter_num > 3 and np.any(fit_failed):
                idx = fit_failed.nonzero()[0]
                for i in idx:  # pylint: disable=not-an-iterable
                    stars.all_stars[i]._excluded_from_fit = True

            # if no star centers have moved by more than pixel accuracy,
            # stop the iteration loop early
            dx_dy = stars.cutout_center_flat - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = stars.cutout_center_flat

            converged = (center_dist_sq.size > 0
                         and np.nanmax(center_dist_sq) < self.center_accuracy_sq)

            if pbar is not None:
                pbar.update()

            if converged:
                break

        if pbar is not None:
            if converged:
                pbar.write(f'EPSFBuilder converged after {iter_num} '
                           f'iterations (of {self.maxiters} maximum '
                           'iterations)')
            pbar.close()

        epsf = self.epsf_class(data=legacy_epsf.data, flux=legacy_epsf.flux,
                        x_0=legacy_epsf.x_0, y_0=legacy_epsf.y_0,
                        oversampling=legacy_epsf.oversampling,
                        fill_value=legacy_epsf.fill_value)

        if self.calibrate_ppe:
            stars = self._apply_ppe_corrections(stars, ppe_map, final=True)

        return epsf, stars


class PPEMap:
    def __init__(self, supersampling, flux_ppe, x_ppe, y_ppe):
        self.supersampling = supersampling
        self.flux_ppe = flux_ppe
        self.x_ppe = x_ppe
        self.y_ppe = y_ppe

    @staticmethod
    def _sample_periodic_map(ppe_map, x, y, supersampling):
        """
        Bilinearly interpolate a periodic supersampled PPE map.
        """
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
        """
        Apply the PPE correction to the input flux and position measurements.

        Parameters
        ----------
        flux : float or array-like
            The flux measurement(s) to correct.

        x : float or array-like
            The x position measurement(s) to correct.

        y : float or array-like
            The y position measurement(s) to correct.

        Returns
        -------
        corrected_flux : float or array-like
            The PPE-corrected flux measurement(s).

        corrected_x : float or array-like
            The PPE-corrected x position measurement(s).

        corrected_y : float or array-like
            The PPE-corrected y position measurement(s).
        """
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
        invalid_flux_scale = (~np.isfinite(flux_scale)) | (flux_scale == 0.0)
        corrected_flux = np.array(flux, copy=True)
        np.divide(flux, flux_scale, out=corrected_flux,
                  where=~invalid_flux_scale)

        x_residual = np.where(np.isfinite(x_residual), x_residual, 0.0)
        y_residual = np.where(np.isfinite(y_residual), y_residual, 0.0)
        corrected_x = x - x_residual
        corrected_y = y - y_residual

        if flux_scalar and x_scalar and y_scalar:
            return corrected_flux.item(), corrected_x.item(), corrected_y.item()

        return corrected_flux, corrected_x, corrected_y
    
    def plot_maps(self):
        """
        Plot the PPE maps for visual inspection.
        """
        import matplotlib.pyplot as plt

        # Define the subpixel x and y positions over which to sample the maps
        sample_factor = 3

        yphase = np.linspace(0.0, 1.0, self.supersampling[0] * sample_factor,
                             endpoint=False)
        xphase = np.linspace(0.0, 1.0, self.supersampling[1] * sample_factor,
                             endpoint=False)
        xx, yy = np.meshgrid(xphase, yphase)

        # Evaluate the flux, x, and y PPE maps at these positions
        flux_map = self._sample_periodic_map(self.flux_ppe, xx, yy,
                                             self.supersampling)
        x_map = self._sample_periodic_map(self.x_ppe, xx, yy,
                                          self.supersampling)
        y_map = self._sample_periodic_map(self.y_ppe, xx, yy,
                                          self.supersampling)

        # Plot the maps as three subplots each with their own colorbar
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

        map_data = (
            (flux_map, 'Flux PPE', 'Fractional flux residual'),
            (x_map, 'X PPE', 'X residual (pixels)'),
            (y_map, 'Y PPE', 'Y residual (pixels)'),
        )
        extent = (0.0, 1.0, 0.0, 1.0)

        for ax, (data, title, cbar_label) in zip(axes, map_data):
            im = ax.imshow(data, origin='lower', extent=extent, cmap='coolwarm',
                           aspect='equal')
            ax.set_title(title)
            ax.set_xlabel('Subpixel x phase')
            ax.set_ylabel('Subpixel y phase')
            fig.colorbar(im, ax=ax, label=cbar_label)

        plt.show()
