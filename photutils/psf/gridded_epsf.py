# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build and fit a gridded ePSF (GEPSF) that accounts
for spatial PSF variations across a detector.
"""

import copy
import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import RectBivariateSpline

from photutils.psf.epsf import EPSFFitter, EPSFBuilder
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.gridded_models import GriddedPSFModel
from photutils.psf.image_models import ImagePSF, _LegacyEPSFModel
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils.cutouts import _overlap_slices as overlap_slices


__all__ = ['GriddedEPSFFitter', 'GriddedEPSFBuilder']


class GriddedEPSFFitter:
    """
    Class to fit a spatially-varying gridded ePSF model to stars.

    The gridded ePSF model is interpolated at each star's position
    using bilinear interpolation, and residuals are computed from
    the interpolated model.

    Parameters
    ----------
    image_shape : tuple of int
        The (height, width) of the image from which stars were extracted.

    grid_shape : tuple of int
        The (n_ygrids, n_xgrids) number of ePSF grid cells to use
        across the image.

    fitter : `EPSFFitter`, optional
        An `EPSFFitter` object to use for fitting. If `None`, then the
        default `EPSFFitter` will be used.

    **fitter_kwargs : dict, optional
        Additional keyword arguments to pass to the `EPSFFitter`.
    """

    def __init__(self, image_shape, grid_shape, *, fitter=None,
                 **fitter_kwargs):
        self.image_shape = np.asarray(image_shape)
        self.grid_shape = np.asarray(grid_shape)

        if fitter is None:
            fitter = EPSFFitter()
        if not isinstance(fitter, EPSFFitter):
            msg = 'fitter must be an EPSFFitter instance'
            raise TypeError(msg)
        self.fitter = fitter

        # Remove fitter kwargs that we manage
        remove_kwargs = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            if kwarg in fitter_kwargs:
                del fitter_kwargs[kwarg]
        self.fitter_kwargs = fitter_kwargs

        # Calculate grid positions and cell boundaries
        self._setup_grid()

    def _setup_grid(self):
        """
        Calculate the grid positions and cell boundaries.
        """
        # Create evenly-spaced grid points across the detector
        y_edges = np.linspace(0, self.image_shape[0], self.grid_shape[0] + 1)
        x_edges = np.linspace(0, self.image_shape[1], self.grid_shape[1] + 1)

        # Grid positions are at cell centers
        self.y_grid = (y_edges[:-1] + y_edges[1:]) / 2.0
        self.x_grid = (x_edges[:-1] + x_edges[1:]) / 2.0

        # Store grid cell edges
        self.y_edges = y_edges
        self.x_edges = x_edges

        # Create all grid positions as (x, y) pairs
        xg, yg = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_xypos = np.column_stack([xg.ravel(), yg.ravel()])

    def _get_interpolated_epsf_at_position(self, gridded_epsf_data,
                                          gridded_epsf_oversampling, x, y):
        """
        Get an interpolated ePSF at a specific (x, y) position using
        bilinear interpolation.

        Parameters
        ----------
        gridded_epsf_data : 3D `~numpy.ndarray`
            The grid of ePSF models with shape (n_psfs, ny, nx).

        gridded_epsf_oversampling : tuple or array_like
            The oversampling factors (y_oversamp, x_oversamp).

        x : float
            The x position in the detector frame.

        y : float
            The y position in the detector frame.

        Returns
        -------
        epsf_data : 2D `~numpy.ndarray`
            The interpolated ePSF at the given position.
        """
        # Find neighboring grid points; allow extrapolation for stars
        # outside the grid by computing weights from unclipped positions.
        x_idx = np.searchsorted(self.x_grid, x)
        y_idx = np.searchsorted(self.y_grid, y)

        # Clamp indices to a valid interpolation cell.
        x_idx = np.clip(x_idx, 0, len(self.x_grid) - 2)
        y_idx = np.clip(y_idx, 0, len(self.y_grid) - 2)

        x0, x1 = self.x_grid[x_idx], self.x_grid[x_idx + 1]
        y0, y1 = self.y_grid[y_idx], self.y_grid[y_idx + 1]

        # Compute bilinear weights (allows extrapolation when x or y is
        # outside the grid range).
        wx1 = (x - x0) / (x1 - x0) if x1 != x0 else 0.5
        wy1 = (y - y0) / (y1 - y0) if y1 != y0 else 0.5
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        # Get the four surrounding ePSF models
        idx_00 = y_idx * len(self.x_grid) + x_idx
        idx_10 = y_idx * len(self.x_grid) + x_idx + 1
        idx_01 = (y_idx + 1) * len(self.x_grid) + x_idx
        idx_11 = (y_idx + 1) * len(self.x_grid) + x_idx + 1

        # Interpolate the ePSF data
        epsf = (wx0 * wy0 * gridded_epsf_data[idx_00] +
                wx1 * wy0 * gridded_epsf_data[idx_10] +
                wx0 * wy1 * gridded_epsf_data[idx_01] +
                wx1 * wy1 * gridded_epsf_data[idx_11])

        return epsf

    def _fit_star_with_gridded_model(self, gridded_epsf_data,
                                    gridded_epsf_oversampling,
                                    star, fit_boxsize):
        """
        Fit a single star using the gridded ePSF model.

        The ePSF is interpolated at the star's position before fitting.
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
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            data = star.data
            weights = star.weights
            x0 = 0
            y0 = 0

        # Get interpolated ePSF at the star's position (in detector coordinates)
        star_x, star_y = star.center
        epsf_data = self._get_interpolated_epsf_at_position(
            gridded_epsf_data, gridded_epsf_oversampling,
            star_x, star_y)

        # Create an ImagePSF from the interpolated ePSF data
        epsf = ImagePSF(epsf_data, oversampling=gridded_epsf_oversampling,
                       fill_value=0.0)
        _epsf = copy.deepcopy(epsf)

        # Define positions in the undersampled grid
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # Set initial guesses
        _epsf.flux = star.flux
        _epsf.x_0 = 0.0
        _epsf.y_0 = 0.0

        try:
            fitted_epsf = self.fitter.fitter(model=_epsf, x=xx, y=yy,
                                             z=data, weights=weights,
                                             **self.fitter_kwargs)
        except Exception as e:
            warnings.warn(f'The star at ({star.center[0]}, '
                          f'{star.center[1]}) could not be fit: {e}',
                          AstropyUserWarning)
            star = copy.deepcopy(star)
            star._fit_error_status = 1
            return star

        # Update star with fitted parameters
        star = copy.deepcopy(star)
        star.flux = fitted_epsf.flux.value
        star._cutout_center = (star.cutout_center[0] + fitted_epsf.x_0.value,
                               star.cutout_center[1] + fitted_epsf.y_0.value)

        if hasattr(self.fitter.fitter, 'fit_info'):
            star._fitinfo = self.fitter.fitter.fit_info

        return star

    def __call__(self, gridded_psf_model, stars):
        """
        Fit the gridded ePSF model to stars.

        Parameters
        ----------
        gridded_psf_model : `GriddedPSFModel`
            The gridded ePSF model.

        stars : `EPSFStars` object
            The stars to be fit.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars with updated centers and fluxes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(gridded_psf_model, GriddedPSFModel):
            msg = 'The input model must be a GriddedPSFModel'
            raise TypeError(msg)

        gridded_epsf_data = gridded_psf_model.data
        gridded_epsf_oversampling = gridded_psf_model.oversampling

        # Fit each star
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                fitted_star = self._fit_star_with_gridded_model(
                    gridded_epsf_data, gridded_epsf_oversampling,
                    star, self.fitter.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    fitted_star.append(
                        self._fit_star_with_gridded_model(
                            gridded_epsf_data, gridded_epsf_oversampling,
                            linked_star, self.fitter.fit_boxsize))
                fitted_star = LinkedEPSFStar(fitted_star)

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)


class GriddedEPSFBuilder:
    """
    Class to build a spatially-varying, gridded ePSF (GEPSF).

    The gridded ePSF is built by dividing the detector into a grid and
    constructing an ePSF model at each grid point. Each ePSF is built
    using stars that fall within that grid cell. The ePSF at each grid
    point is improved iteratively.

    Parameters
    ----------
    image_shape : tuple of int
        The (height, width) of the image from which stars were extracted.

    grid_shape : tuple of int
        The (n_ygrids, n_xgrids) number of ePSF grid cells to use
        across the image.

    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input stars. If scalar, used for both axes. If array, must be
        (y, x) order.

    shape : float, tuple of two floats, or `None`, optional
        The shape of each output ePSF in the grid. If `None`, derived
        from the input stars and oversampling factor.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF at each grid point.

    residual_smoothing_kernel : {'gaussian'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF residuals at each
        grid point.

    recentering_func : callable, optional
        A callable to calculate the centroid of a 2D array.

    recentering_maxiters : int, optional
        Maximum number of recentering iterations per build iteration.

    fitter : `EPSFFitter` object, optional
        An `EPSFFitter` to use for fitting. If `None`, default
        `EPSFFitter` is used.

    maxiters : int, optional
        The maximum number of iterations to perform.

    progress_bar : bool, optional
        Whether to print progress bars during build iterations.

    normalise_epsf : bool, optional
        If `True`, each ePSF is normalized after each iteration.

    norm_radius : float, optional
        The pixel radius over which each ePSF is normalized.

    min_stars_per_gridcell : int, optional
        Minimum number of stars required in a grid cell to build an
        ePSF at that location. Cells with fewer stars will not be
        updated. Default is 5.

    **epsf_builder_kwargs : dict, optional
        Additional keyword arguments to pass to the `EPSFBuilder`
        when building individual grid cell ePSFs.
    """

    def __init__(self, image_shape, grid_shape, oversampling, *,
                 shape=None,
                 smoothing_kernel='quartic',
                 residual_smoothing_kernel='gaussian',
                 recentering_func=None,
                 recentering_maxiters=20,
                 fitter=None,
                 maxiters=10,
                 progress_bar=True,
                 normalise_epsf=True,
                 norm_radius=5,
                 min_stars_per_gridcell=5,
                 **epsf_builder_kwargs):

        self.image_shape = np.asarray(image_shape)
        self.grid_shape = np.asarray(grid_shape)
        self.oversampling = as_pair('oversampling', oversampling,
                                   lower_bound=(0, 1))
        self.shape = shape
        self.smoothing_kernel = smoothing_kernel
        self.residual_smoothing_kernel = residual_smoothing_kernel
        self.recentering_maxiters = recentering_maxiters
        self.maxiters = maxiters
        self.progress_bar = progress_bar
        self.normalise_epsf = normalise_epsf
        self.norm_radius = norm_radius
        self.min_stars_per_gridcell = int(min_stars_per_gridcell)

        if recentering_func is None:
            from photutils.centroids import centroid_com
            recentering_func = centroid_com
        self.recentering_func = recentering_func

        # Setup the fitter
        self.gridded_fitter = GriddedEPSFFitter(
            image_shape, grid_shape, fitter=fitter)

        # Store ePSFBuilder kwargs
        self.epsf_builder_kwargs = copy.deepcopy(epsf_builder_kwargs)

        # Setup grid
        self._setup_grid()

        # Create individual EPSFBuilders for each grid cell
        self._create_grid_cell_builders()

    def _setup_grid(self):
        """
        Calculate grid positions and cell boundaries.
        """
        y_edges = np.linspace(0, self.image_shape[0], self.grid_shape[0] + 1)
        x_edges = np.linspace(0, self.image_shape[1], self.grid_shape[1] + 1)

        self.y_grid = (y_edges[:-1] + y_edges[1:]) / 2.0
        self.x_grid = (x_edges[:-1] + x_edges[1:]) / 2.0
        self.y_edges = y_edges
        self.x_edges = x_edges

        xg, yg = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_xypos = np.column_stack([xg.ravel(), yg.ravel()])

    def _create_grid_cell_builders(self):
        """
        Create an individual EPSFBuilder for each grid cell.
        """
        self.grid_builders = []
        for i in range(len(self.y_grid)):
            row = []
            for j in range(len(self.x_grid)):
                builder = EPSFBuilder(
                    oversampling=self.oversampling,
                    shape=self.shape,
                    smoothing_kernel=self.smoothing_kernel,
                    residual_smoothing_kernel=self.residual_smoothing_kernel,
                    recentering_func=self.recentering_func,
                    recentering_maxiters=self.recentering_maxiters,
                    maxiters=self.maxiters,
                    progress_bar=False,  # We'll manage progress globally
                    normalise_epsf=self.normalise_epsf,
                    norm_radius=self.norm_radius,
                    **self.epsf_builder_kwargs)
                row.append(builder)
            self.grid_builders.append(row)

    def _partition_stars_by_gridcell(self, stars):
        """
        Partition stars into grid cells based on their positions.

        Returns
        -------
        grid_stars : list of list of EPSFStars
            2D list where grid_stars[i][j] contains the stars in
            grid cell (i, j).
        """
        grid_stars = []
        for i in range(len(self.y_grid)):
            row = []
            for j in range(len(self.x_grid)):
                y_min, y_max = self.y_edges[i], self.y_edges[i + 1]
                x_min, x_max = self.x_edges[j], self.x_edges[j + 1]

                # Select stars in this grid cell
                cell_stars = []
                for star in stars:
                    star_x, star_y = star.center
                    if (x_min <= star_x < x_max and y_min <= star_y < y_max):
                        cell_stars.append(star)

                if len(cell_stars) >= self.min_stars_per_gridcell:
                    row.append(EPSFStars(cell_stars))
                else:
                    row.append(None)  # Not enough stars for this cell

            grid_stars.append(row)

        return grid_stars

    def build_epsf(self, stars):
        """
        Build a gridded ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF. These should have origins
            set so that their center positions are in detector coordinates.

        Returns
        -------
        gridded_epsf_model : `GriddedPSFModel`
            The constructed gridded ePSF model.

        fitted_stars : `EPSFStars` object
            The input stars with updated centers and fluxes derived
            from fitting the gridded ePSF.
        """
        # Partition stars into grid cells
        grid_stars = self._partition_stars_by_gridcell(stars)

        # Build initial ePSF for each grid cell
        grid_epsfs = []
        for i, row in enumerate(grid_stars):
            for j, cell_stars in enumerate(row):
                if cell_stars is None:
                    grid_epsfs.append(None)
                else:
                    builder = self.grid_builders[i][j]
                    epsf, _ = builder.build_epsf(cell_stars)
                    grid_epsfs.append(epsf)

        # Create initial GriddedPSFModel
        gridded_model = self._make_gridded_psf_model(grid_epsfs)

        # Iteratively improve the gridded ePSF
        pbar = None
        if self.progress_bar:
            pbar = add_progress_bar(total=self.maxiters,
                                   desc='GriddedEPSFBuilder')

        for iteration in range(self.maxiters):
            # Fit gridded model to all stars
            fitted_stars = self.gridded_fitter(gridded_model, stars)

            # Compute residuals and update each grid cell
            grid_stars = self._partition_stars_by_gridcell(fitted_stars)
            new_grid_epsfs = []

            for i, row in enumerate(grid_stars):
                for j, cell_stars in enumerate(row):
                    if cell_stars is None:
                        # Use previous ePSF if not enough stars
                        new_grid_epsfs.append(grid_epsfs[i * len(self.x_grid) + j])
                    else:
                        builder = self.grid_builders[i][j]
                        init_epsf = grid_epsfs[i * len(self.x_grid) + j]
                        try:
                            epsf, _ = builder.build_epsf(
                                cell_stars, init_epsf=init_epsf)
                            new_grid_epsfs.append(epsf)
                        except Exception as e:
                            warnings.warn(
                                f'Failed to update ePSF at grid cell ({i}, {j}): {e}',
                                AstropyUserWarning)
                            new_grid_epsfs.append(
                                grid_epsfs[i * len(self.x_grid) + j])

            grid_epsfs = new_grid_epsfs
            gridded_model = self._make_gridded_psf_model(grid_epsfs)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Final fit to get improved star parameters
        fitted_stars = self.gridded_fitter(gridded_model, stars)

        return gridded_model, fitted_stars

    def _make_gridded_psf_model(self, grid_epsfs):
        """
        Create a GriddedPSFModel from a list of ImagePSF objects.

        Parameters
        ----------
        grid_epsfs : list of ImagePSF or None
            The ePSF models for each grid cell. None means use a default.

        Returns
        -------
        gridded_model : `GriddedPSFModel`
            The gridded PSF model.
        """
        # Filter out None and create 3D data array
        valid_epsfs = []
        for epsf in grid_epsfs:
            if epsf is None:
                valid_epsfs.append(np.zeros_like(grid_epsfs[0].data))
            else:
                valid_epsfs.append(epsf.data)

        # Reshape to match grid layout, then flatten
        grid_data = np.array(valid_epsfs)

        # Create NDData for GriddedPSFModel
        nddata = NDData(
            data=grid_data,
            meta={
                'grid_xypos': self.grid_xypos,
                'oversampling': self.oversampling.tolist()
            }
        )

        return GriddedPSFModel(nddata)

    def __call__(self, stars):
        """
        Build a gridded ePSF from stars.

        Equivalent to calling build_epsf.
        """
        return self.build_epsf(stars)
