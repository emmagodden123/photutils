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

    def _get_interpolated_epsf_at_position(self, gridded_psf_model, x, y):
        """
        Get an interpolated ePSF at a specific (x, y) position using
        bilinear interpolation.

        Parameters
        ----------
        gridded_psf_model : `GriddedPSFModel`
            The gridded ePSF model.

        x : float
            The x position in the detector frame.

        y : float
            The y position in the detector frame.

        Returns
        -------
        epsf_data : 2D `~numpy.ndarray`
            The interpolated ePSF at the given position.
        """
        data = gridded_psf_model.data
        if data.shape[0] == 1:
            return data[0]

        # Use GriddedPSFModel interpolation geometry/edge behavior.
        grid_idx, grid_xy = gridded_psf_model._find_bounding_points(x, y)
        weights = gridded_psf_model._calc_bilinear_weights(x, y, grid_xy)

        epsf = np.zeros_like(data[grid_idx[0]], dtype=float)
        for idx, weight in zip(grid_idx, weights, strict=True):
            if weight != 0.0:
                epsf += weight * data[idx]

        return epsf

    def _fit_star_with_gridded_model(self, gridded_psf_model,
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
            gridded_psf_model, star_x, star_y)

        # Create an ImagePSF from the interpolated ePSF data
        epsf = ImagePSF(epsf_data, oversampling=gridded_psf_model.oversampling,
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
        except TypeError:
            fitted_epsf = self.fitter.fitter(model=_epsf, x=xx, y=yy,
                                             z=data,
                                             **self.fitter_kwargs)
        except Exception as e:
            warnings.warn(f'The star at ({star.center[0]}, '
                          f'{star.center[1]}) could not be fit: {e}',
                          AstropyUserWarning)
            star = copy.deepcopy(star)
            star._fit_error_status = 1
            return star

        fit_error_status = 0
        fit_info = None
        if hasattr(self.fitter.fitter, 'fit_info'):
            fit_info = copy.copy(self.fitter.fitter.fit_info)
            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2

        if fit_error_status == 2:
            star = copy.deepcopy(star)
            star._fit_error_status = fit_error_status
            return star

        # Update star with fitted parameters
        star = copy.deepcopy(star)
        star.flux = fitted_epsf.flux.value
        star.cutout_center = (star.cutout_center[0] + fitted_epsf.x_0.value,
                              star.cutout_center[1] + fitted_epsf.y_0.value)
        star._fit_error_status = 0
        star._fit_info = fit_info
        star._fitinfo = fit_info

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

        # Fit each star
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                fitted_star = self._fit_star_with_gridded_model(
                    gridded_psf_model, star, self.fitter.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    fitted_star.append(
                        self._fit_star_with_gridded_model(
                            gridded_psf_model, linked_star,
                            self.fitter.fit_boxsize))
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

    recenter_epsf : bool, optional
        Whether to recenter each cell ePSF during its build iterations.

    fitter : `EPSFFitter` object, optional
        An `EPSFFitter` to use for fitting. If `None`, default
        `EPSFFitter` is used.

    maxiters : int, optional
        The maximum number of iterations to perform.

    progress_bar : bool, optional
        Whether to print progress bars during build iterations.

    normalise_epsf : bool, optional
        If `True`, each ePSF is normalized after each iteration.

    min_stars_per_gridcell : int, optional
        Minimum number of stars required in a grid cell to build an
        ePSF at that location. Cells with fewer stars will not be
        updated. Default is 5.

    plot_diagnostics : bool, optional
        Whether to show diagnostic plots during gridded ePSF building.
        If `True`, diagnostic plots are shown for sample distributions,
        gridded flux PPE maps, and gridded ePSF models.

    **epsf_builder_kwargs : dict, optional
        Additional keyword arguments to pass to the `EPSFBuilder`
        when building individual grid cell ePSFs.
    """

    def __init__(self, image_shape, grid_shape, oversampling, *,
                 fitter=None,
                 maxiters=5,
                 progress_bar=True,
                 min_stars_per_gridcell=5,
                 plot_diagnostics=False,
                 epsf_builder_kwargs=None):

        self.image_shape = np.asarray(image_shape)
        self.grid_shape = np.asarray(grid_shape)
        self.oversampling = as_pair('oversampling', oversampling,
                                   lower_bound=(0, 1))
        self.maxiters = maxiters
        self.progress_bar = progress_bar
        self.min_stars_per_gridcell = int(min_stars_per_gridcell)
        self.plot_diagnostics = bool(plot_diagnostics)

        # Setup the fitter
        self.fitter = fitter
        self.gridded_fitter = GriddedEPSFFitter(
            image_shape, grid_shape, fitter=fitter)

        # Store ePSFBuilder kwargs
        self.epsf_builder_kwargs = copy.deepcopy(epsf_builder_kwargs or {})

        # Setup grid
        self._setup_grid()

        # Create individual EPSFBuilders for each grid cell
        self._create_grid_cell_builders()

    def _get_matplotlib(self):
        """
        Import and return matplotlib.pyplot.

        Returns
        -------
        plt : module or `None`
            The matplotlib pyplot module. If unavailable, `None` is
            returned and a warning is emitted.
        """
        if not self.plot_diagnostics:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('plot_diagnostics=True but matplotlib is not '
                          'available. Skipping diagnostic plots.',
                          AstropyUserWarning)
            return None

        return plt

    def _get_grid_indices(self, x, y):
        """
        Return the `(iy, ix)` grid-cell index for detector coordinates.
        """
        if (x < self.x_edges[0] or x > self.x_edges[-1]
                or y < self.y_edges[0] or y > self.y_edges[-1]):
            return None

        ix = np.searchsorted(self.x_edges, x, side='right') - 1
        iy = np.searchsorted(self.y_edges, y, side='right') - 1

        ix = int(np.clip(ix, 0, len(self.x_grid) - 1))
        iy = int(np.clip(iy, 0, len(self.y_grid) - 1))

        return iy, ix

    def _iter_star_samples(self, stars):
        """
        Iterate through all stars and return samples with linked IDs.

        Yields
        ------
        sample : dict
            A dictionary containing star information including detector
            position, subpixel phase, flux, and linked-star ID.
        """
        unlinked_id = -1

        for group_id, star_group in enumerate(stars):
            if isinstance(star_group, LinkedEPSFStar):
                for linked_star in star_group:
                    x, y = linked_star.center
                    grid_idx = self._get_grid_indices(x, y)
                    if grid_idx is None:
                        continue
                    yield {
                        'x': x,
                        'y': y,
                        'flux': linked_star.flux,
                        'subpixel_x': np.mod(x, 1.0),
                        'subpixel_y': np.mod(y, 1.0),
                        'grid_iy': grid_idx[0],
                        'grid_ix': grid_idx[1],
                        'linked_id': group_id,
                    }
            elif isinstance(star_group, EPSFStar):
                x, y = star_group.center
                grid_idx = self._get_grid_indices(x, y)
                if grid_idx is None:
                    continue
                yield {
                    'x': x,
                    'y': y,
                    'flux': star_group.flux,
                    'subpixel_x': np.mod(x, 1.0),
                    'subpixel_y': np.mod(y, 1.0),
                    'grid_iy': grid_idx[0],
                    'grid_ix': grid_idx[1],
                    'linked_id': unlinked_id,
                }
                unlinked_id -= 1

    def _collect_sample_arrays(self, stars):
        """
        Collect star-sample arrays used by diagnostics.
        """
        samples = list(self._iter_star_samples(stars))
        if len(samples) == 0:
            return None

        sample_arrays = {
            'x': np.asarray([item['x'] for item in samples], dtype=float),
            'y': np.asarray([item['y'] for item in samples], dtype=float),
            'flux': np.asarray([item['flux'] for item in samples], dtype=float),
            'subpixel_x': np.asarray([item['subpixel_x'] for item in samples],
                                     dtype=float),
            'subpixel_y': np.asarray([item['subpixel_y'] for item in samples],
                                     dtype=float),
            'grid_iy': np.asarray([item['grid_iy'] for item in samples],
                                  dtype=int),
            'grid_ix': np.asarray([item['grid_ix'] for item in samples],
                                  dtype=int),
            'linked_id': np.asarray([item['linked_id'] for item in samples],
                                    dtype=int),
        }

        return sample_arrays

    def _compute_subpixel_counts(self, sample_arrays):
        """
        Compute per-cell subpixel sample counts and linked-star counts.
        """
        ny_grid = len(self.y_grid)
        nx_grid = len(self.x_grid)
        osy, osx = self.oversampling

        sample_counts = np.zeros((ny_grid, nx_grid, osy, osx), dtype=int)
        unique_linked_counts = np.zeros((ny_grid, nx_grid, osy, osx),
                                        dtype=int)

        unique_sets = [[[[set() for _ in range(osx)] for _ in range(osy)]
                        for _ in range(nx_grid)] for _ in range(ny_grid)]

        xbin = np.floor(sample_arrays['subpixel_x'] * osx).astype(int)
        ybin = np.floor(sample_arrays['subpixel_y'] * osy).astype(int)
        xbin = np.clip(xbin, 0, osx - 1)
        ybin = np.clip(ybin, 0, osy - 1)

        for i in range(sample_arrays['x'].size):
            iy = sample_arrays['grid_iy'][i]
            ix = sample_arrays['grid_ix'][i]
            xb = xbin[i]
            yb = ybin[i]
            sample_counts[iy, ix, yb, xb] += 1
            linked_id = sample_arrays['linked_id'][i]
            if linked_id >= 0:
                unique_sets[iy][ix][yb][xb].add(linked_id)

        for iy in range(ny_grid):
            for ix in range(nx_grid):
                for yb in range(osy):
                    for xb in range(osx):
                        unique_linked_counts[iy, ix, yb, xb] = len(
                            unique_sets[iy][ix][yb][xb])

        return sample_counts, unique_linked_counts

    def _compute_flux_ppe_grid_maps(self, sample_arrays):
        """
        Compute a flux PPE map per image grid cell.

        Linked stars use measured-minus-linked-mean fractional flux
        residuals when available. Unlinked stars (or linked stars with
        undefined means) use measured-minus-cell-median fractional flux
        residuals.
        """
        ny_grid = len(self.y_grid)
        nx_grid = len(self.x_grid)
        osy, osx = self.oversampling

        flux_maps = np.full((ny_grid, nx_grid, osy, osx), np.nan, dtype=float)

        linked_id = sample_arrays['linked_id']
        flux = sample_arrays['flux']

        linked_mean_flux = {}
        for lid in np.unique(linked_id[linked_id >= 0]):
            mask = (linked_id == lid) & np.isfinite(flux)
            if np.any(mask):
                linked_mean_flux[lid] = np.nanmedian(flux[mask])

        cell_median_flux = np.full((ny_grid, nx_grid), np.nan, dtype=float)
        for iy in range(ny_grid):
            for ix in range(nx_grid):
                mask = ((sample_arrays['grid_iy'] == iy)
                        & (sample_arrays['grid_ix'] == ix)
                        & np.isfinite(flux))
                if np.any(mask):
                    cell_median_flux[iy, ix] = np.nanmedian(flux[mask])

        flux_residual = np.zeros_like(flux, dtype=float)
        for i in range(flux.size):
            this_flux = flux[i]
            if not np.isfinite(this_flux):
                flux_residual[i] = np.nan
                continue

            lid = linked_id[i]
            if lid >= 0 and lid in linked_mean_flux:
                mean_flux = linked_mean_flux[lid]
            else:
                mean_flux = cell_median_flux[sample_arrays['grid_iy'][i],
                                             sample_arrays['grid_ix'][i]]

            if np.isfinite(mean_flux) and mean_flux != 0.0:
                flux_residual[i] = (this_flux - mean_flux) / mean_flux
            else:
                flux_residual[i] = 0.0

        xbin = np.floor(sample_arrays['subpixel_x'] * osx).astype(int)
        ybin = np.floor(sample_arrays['subpixel_y'] * osy).astype(int)
        xbin = np.clip(xbin, 0, osx - 1)
        ybin = np.clip(ybin, 0, osy - 1)

        for iy in range(ny_grid):
            for ix in range(nx_grid):
                cell_mask = ((sample_arrays['grid_iy'] == iy)
                             & (sample_arrays['grid_ix'] == ix))
                if not np.any(cell_mask):
                    flux_maps[iy, ix, :, :] = 0.0
                    continue

                for yb in range(osy):
                    for xb in range(osx):
                        bin_mask = (cell_mask & (ybin == yb) & (xbin == xb)
                                    & np.isfinite(flux_residual))
                        if np.any(bin_mask):
                            flux_maps[iy, ix, yb, xb] = np.nanmedian(
                                flux_residual[bin_mask])

                missing = ~np.isfinite(flux_maps[iy, ix, :, :])
                if np.all(missing):
                    flux_maps[iy, ix, :, :] = 0.0
                else:
                    flux_maps[iy, ix, :, :][missing] = 0.0

        return flux_maps

    def _plot_star_distribution_diagnostics(self, sample_arrays, stage_label):
        """
        Plot star sample distribution and per-cell subpixel occupancy.
        """
        plt = self._get_matplotlib()
        if plt is None or sample_arrays is None:
            return

        sample_counts, unique_linked_counts = self._compute_subpixel_counts(
            sample_arrays)

        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.scatter(sample_arrays['x'], sample_arrays['y'], s=12, alpha=0.7)
        for xedge in self.x_edges:
            ax.axvline(xedge, color='k', linewidth=0.7, alpha=0.5)
        for yedge in self.y_edges:
            ax.axhline(yedge, color='k', linewidth=0.7, alpha=0.5)
        ax.set_xlim(0, self.image_shape[1])
        ax.set_ylim(0, self.image_shape[0])
        ax.set_xlabel('Detector x')
        ax.set_ylabel('Detector y')
        ax.set_title(f'Star sample distribution ({stage_label})')
        plt.show()

        nrows = len(self.y_grid)
        ncols = len(self.x_grid)
        fig, axes = plt.subplots(2 * nrows, ncols,
                                 figsize=(4.2 * ncols, 5.0 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)

        for iy in range(nrows):
            for ix in range(ncols):
                top_ax = axes[2 * iy, ix]
                bot_ax = axes[2 * iy + 1, ix]

                counts = sample_counts[iy, ix, :, :]
                unique_counts = unique_linked_counts[iy, ix, :, :]
                linked_ids = sample_arrays['linked_id']
                cell_mask = ((sample_arrays['grid_iy'] == iy)
                             & (sample_arrays['grid_ix'] == ix)
                             & (linked_ids >= 0))
                n_unique_linked = np.unique(linked_ids[cell_mask]).size

                im0 = top_ax.imshow(counts, origin='lower', cmap='viridis',
                                    extent=(0.0, 1.0, 0.0, 1.0),
                                    aspect='equal')
                top_ax.set_title(f'Cell ({iy}, {ix}) samples: {counts.sum()}')
                top_ax.set_xlabel('Subpixel x phase')
                top_ax.set_ylabel('Subpixel y phase')
                fig.colorbar(im0, ax=top_ax, fraction=0.046, pad=0.04,
                             label='Sample count')

                im1 = bot_ax.imshow(unique_counts, origin='lower',
                                    cmap='magma', extent=(0.0, 1.0, 0.0, 1.0),
                                    aspect='equal')
                bot_ax.set_title('Unique linked stars per bin: '
                                 f'{n_unique_linked}')
                bot_ax.set_xlabel('Subpixel x phase')
                bot_ax.set_ylabel('Subpixel y phase')
                fig.colorbar(im1, ax=bot_ax, fraction=0.046, pad=0.04,
                             label='Unique linked-star count')

        fig.suptitle(f'Subpixel occupancy by grid cell ({stage_label})')
        plt.show()

    def _plot_flux_ppe_grid_maps(self, sample_arrays, stage_label):
        """
        Plot per-cell flux PPE maps.
        """
        plt = self._get_matplotlib()
        if plt is None or sample_arrays is None:
            return

        flux_maps = self._compute_flux_ppe_grid_maps(sample_arrays)
        nrows = len(self.y_grid)
        ncols = len(self.x_grid)

        finite = np.isfinite(flux_maps)
        if np.any(finite):
            vmax = np.nanmax(np.abs(flux_maps[finite]))
            vmax = vmax if vmax > 0 else 1.0e-12
        else:
            vmax = 1.0e-12

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.2 * ncols, 3.8 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)

        for iy in range(nrows):
            for ix in range(ncols):
                ax = axes[iy, ix]
                im = ax.imshow(flux_maps[iy, ix, :, :], origin='lower',
                               cmap='coolwarm', vmin=-vmax, vmax=vmax,
                               extent=(0.0, 1.0, 0.0, 1.0), aspect='equal')
                nsamp = np.count_nonzero((sample_arrays['grid_iy'] == iy)
                                         & (sample_arrays['grid_ix'] == ix))
                ax.set_title(f'Cell ({iy}, {ix}) N={nsamp}')
                ax.set_xlabel('Subpixel x phase')
                ax.set_ylabel('Subpixel y phase')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label='Flux PPE')

        fig.suptitle(f'Gridded flux PPE maps ({stage_label})')
        plt.show()

    def _plot_grid_epsf_models(self, grid_epsfs, stage_label):
        """
        Plot ePSF models for all image grid cells.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return

        nrows = len(self.y_grid)
        ncols = len(self.x_grid)

        epsf_data = [epsf.data for epsf in grid_epsfs if epsf is not None]
        if len(epsf_data) > 0:
            finite_vals = np.concatenate([arr[np.isfinite(arr)]
                                          for arr in epsf_data
                                          if np.any(np.isfinite(arr))])
            if finite_vals.size > 0:
                vmin = np.nanmin(finite_vals)
                vmax = np.nanmax(finite_vals)
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.2 * ncols, 3.8 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)

        for iy in range(nrows):
            for ix in range(ncols):
                ax = axes[iy, ix]
                idx = iy * ncols + ix
                epsf = grid_epsfs[idx]

                if epsf is None:
                    data = np.zeros((3, 3), dtype=float)
                    title = f'Cell ({iy}, {ix}) [None]'
                else:
                    data = epsf.data
                    title = f'Cell ({iy}, {ix})'

                im = ax.imshow(data, origin='lower', cmap='viridis',
                               vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title)
                ax.set_xlabel('x (oversampled)')
                ax.set_ylabel('y (oversampled)')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f'Gridded ePSF models ({stage_label})')
        plt.show()

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
                    fitter=self.fitter,
                    maxiters=self.maxiters,
                    progress_bar=False,  # Managed by GriddedEPSFBuilder
                    plot_diagnostics=False,  # We'll manage diagnostics globally
                    **self.epsf_builder_kwargs)
                row.append(builder)
            self.grid_builders.append(row)

    @staticmethod
    def _build_grid_cell_epsf(builder, stars, *, init_model=None,
                              maxiters=None):
        """
        Build or update one grid-cell ePSF.
        """
        if maxiters is None:
            return builder.build_epsf(stars, init_model=init_model)

        builder_maxiters = builder.maxiters
        builder.maxiters = maxiters
        try:
            return builder.build_epsf(stars, init_model=init_model)
        finally:
            builder.maxiters = builder_maxiters

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
                for star_group in stars:
                    if isinstance(star_group, LinkedEPSFStar):
                        # Preserve linked-star groupings within each cell so
                        # the per-cell EPSFBuilder can still apply its linked
                        # flux/position constraints and PPE calibration.
                        linked_cell_stars = []
                        for star in star_group:
                            star_x, star_y = star.center
                            if (x_min <= star_x < x_max
                                    and y_min <= star_y < y_max):
                                linked_cell_stars.append(star)

                        if len(linked_cell_stars) > 1:
                            cell_stars.append(
                                LinkedEPSFStar(linked_cell_stars))
                        elif len(linked_cell_stars) == 1:
                            cell_stars.append(linked_cell_stars[0])
                    elif isinstance(star_group, EPSFStar):
                        # Handle regular EPSFStar
                        star_x, star_y = star_group.center
                        if (x_min <= star_x < x_max and y_min <= star_y < y_max):
                            cell_stars.append(star_group)

                epsf_stars = EPSFStars(cell_stars)
                if epsf_stars.n_all_stars >= self.min_stars_per_gridcell:
                    row.append(epsf_stars)
                else:
                    row.append(None)  # Not enough stars for this cell

            grid_stars.append(row)

        return grid_stars

    def build_epsf(self, stars, init_model=None):
        """
        Build a gridded ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF. These should have origins
            set so that their center positions are in detector coordinates.

        init_model : `GriddedPSFModel`, optional
            The initial gridded ePSF model. If input, it is used as the
            starting model for the iterative build updates.

        Returns
        -------
        gridded_epsf_model : `GriddedPSFModel`
            The constructed gridded ePSF model.

        fitted_stars : `EPSFStars` object
            The input stars with updated centers and fluxes derived
            from fitting the gridded ePSF.
        """
        n_grid = len(self.grid_xypos)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if self.plot_diagnostics:
            initial_samples = self._collect_sample_arrays(stars)
            self._plot_star_distribution_diagnostics(initial_samples,
                                                     'initial samples')
            self._plot_flux_ppe_grid_maps(initial_samples,
                                          'initial samples')

        if init_model is not None:
            if not isinstance(init_model, GriddedPSFModel):
                msg = 'init_model must be a GriddedPSFModel'
                raise TypeError(msg)

            if init_model.data.shape[0] != n_grid:
                msg = (f'init_model must contain {n_grid} grid PSFs in the '
                       'first axis')
                raise ValueError(msg)

            if init_model.grid_xypos.shape != self.grid_xypos.shape:
                msg = 'init_model grid_xypos shape must match builder grid'
                raise ValueError(msg)

            if not np.allclose(init_model.grid_xypos, self.grid_xypos):
                msg = ('init_model grid_xypos must match the builder grid '
                       'positions and ordering')
                raise ValueError(msg)

            gridded_model = init_model
            grid_epsfs = [
                ImagePSF(init_model.data[idx],
                         oversampling=init_model.oversampling,
                         fill_value=0.0)
                for idx in range(n_grid)
            ]
        else:
            # Partition stars into grid cells
            grid_stars = self._partition_stars_by_gridcell(stars)

            # Build initial ePSF for each grid cell
            grid_epsfs = []
            initial_indices = range(n_grid)
            if self.progress_bar and tqdm is not None:
                initial_indices = tqdm(
                    initial_indices,
                    desc='GriddedEPSFBuilder initial grid build',
                    total=n_grid)

            for idx in initial_indices:
                i, j = divmod(idx, len(self.x_grid))
                cell_stars = grid_stars[i][j]

                if cell_stars is None:
                    grid_epsfs.append(None)
                else:
                    builder = self.grid_builders[i][j]
                    epsf, _ = self._build_grid_cell_epsf(
                        builder, cell_stars)
                    grid_epsfs.append(epsf)

            # Create initial GriddedPSFModel
            gridded_model = self._make_gridded_psf_model(grid_epsfs)

        if self.plot_diagnostics:
            self._plot_grid_epsf_models(grid_epsfs, 'initial grid ePSF models')

        # Iteratively improve the gridded ePSF
        for iteration in range(self.maxiters):
            # Fit gridded model to all stars
            fitted_stars = self.gridded_fitter(gridded_model, stars)

            if self.plot_diagnostics:
                iter_samples = self._collect_sample_arrays(fitted_stars)
                stage = f'iteration {iteration + 1}'
                self._plot_flux_ppe_grid_maps(iter_samples, stage)

            # Compute residuals and update each grid cell
            grid_stars = self._partition_stars_by_gridcell(fitted_stars)
            new_grid_epsfs = [None] * n_grid
            cell_indices = range(n_grid)
            if self.progress_bar and tqdm is not None:
                cell_indices = tqdm(
                    cell_indices,
                    desc=(f'GriddedEPSFBuilder iteration '
                          f'{iteration + 1}/{self.maxiters}'),
                    total=n_grid)

            for idx in cell_indices:
                i, j = divmod(idx, len(self.x_grid))
                cell_stars = grid_stars[i][j]

                if cell_stars is None:
                    # Use previous ePSF if not enough stars
                    new_grid_epsfs[idx] = grid_epsfs[idx]
                else:
                    builder = self.grid_builders[i][j]
                    init_epsf = grid_epsfs[idx]
                    try:
                        epsf, _ = self._build_grid_cell_epsf(
                            builder, cell_stars, init_model=init_epsf,
                            maxiters=1)
                        new_grid_epsfs[idx] = epsf
                    except Exception as e:
                        warnings.warn(
                            f'Failed to update ePSF at grid cell ({i}, {j}): {e}',
                            AstropyUserWarning)
                        new_grid_epsfs[idx] = grid_epsfs[idx]

            grid_epsfs = new_grid_epsfs
            gridded_model = self._make_gridded_psf_model(grid_epsfs)

            if self.plot_diagnostics:
                self._plot_grid_epsf_models(
                    grid_epsfs, f'iteration {iteration + 1}')

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
        template_epsf = next((epsf for epsf in grid_epsfs if epsf is not None),
                             None)
        if template_epsf is None:
            msg = ('Cannot create GriddedPSFModel because no grid cells have '
                   'a valid ePSF. Provide init_model or lower '
                   'min_stars_per_gridcell.')
            raise ValueError(msg)

        valid_indices = [idx for idx, epsf in enumerate(grid_epsfs)
                         if epsf is not None]
        valid_xypos = self.grid_xypos[valid_indices]

        valid_epsfs = []
        missing_indices = []
        for epsf in grid_epsfs:
            if epsf is None:
                missing_indices.append(len(valid_epsfs))
                valid_epsfs.append(None)
            else:
                valid_epsfs.append(epsf.data)

        for idx in missing_indices:
            distance_sq = np.sum((valid_xypos - self.grid_xypos[idx])**2,
                                 axis=1)
            nearest_idx = valid_indices[np.argmin(distance_sq)]
            valid_epsfs[idx] = grid_epsfs[nearest_idx].data

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
