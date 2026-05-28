# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Experimental variable-dependency ePSF classes.

This module extends the spatial ePSF prototype by allowing configurable
model dependencies on detector position and/or star flux.
"""

import copy
import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve

from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.spatial_epsf import SpatialEPSFBuilder, SpatialEPSFFitter
from photutils.psf.spatial_epsf import SpatialEPSFModel
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround

__all__ = ['VariableEPSFModel', 'VariableEPSFFitter', 'VariableEPSFBuilder']


def _parse_dependencies(dependencies):
    """
    Parse model dependency labels.
    """
    if isinstance(dependencies, str):
        dependencies = (dependencies,)

    try:
        dependencies = tuple(dependencies)
    except TypeError as exc:
        raise TypeError('dependencies must be an iterable containing '
                        "'Position' and/or 'Flux'") from exc

    allowed = {'Position', 'Flux'}
    invalid = [item for item in dependencies if item not in allowed]
    if invalid:
        raise ValueError("dependencies entries must be 'Position' and/or "
                         "'Flux'")
    return tuple(dict.fromkeys(dependencies))


class VariableEPSFModel(SpatialEPSFModel):
    """
    A configurable dependency ePSF model.

    Parameters
    ----------
    dependencies : iterable of {'Position', 'Flux'}, optional
        Model dependencies. If ``'Position'`` is present, the detector
        position polynomial basis is used. If ``'Flux'`` is present, an
        additive polynomial-in-flux correction image is added.

    flux_coeff_data : 3D `~numpy.ndarray`, optional
        Flux coefficient array with shape ``(n_flux_basis, ny, nx)``.

    flux_degree : int, optional
        Polynomial degree used for the flux basis. The number of basis
        terms is ``flux_degree + 1``.

    flux_reference, flux_scale : float, optional
        Parameters used to normalize flux values via
        ``(flux - flux_reference) / flux_scale``.
    """

    def __init__(self, coeff_data, *, oversampling, detector_shape,
                 degree=1, origin=None, fill_value=0.0,
                 detector_origin=None, detector_span=None,
                 normalize_local_epsf=True, trust_map=None,
                 epsf_class=ImagePSF, dependencies=('Position',),
                 flux_coeff_data=None, flux_degree=1,
                 flux_reference=1.0, flux_scale=1.0):
        dependencies = _parse_dependencies(dependencies)
        if 'Position' not in dependencies and degree != 0:
            warnings.warn('Position dependency is disabled; forcing degree=0 '
                          'for the baseline coefficient model.',
                          AstropyUserWarning)
            degree = 0

        super().__init__(
            coeff_data, oversampling=oversampling,
            detector_shape=detector_shape, degree=degree,
            origin=origin, fill_value=fill_value,
            detector_origin=detector_origin, detector_span=detector_span,
            normalize_local_epsf=normalize_local_epsf, trust_map=trust_map,
            epsf_class=epsf_class)

        self.dependencies = dependencies
        self.flux_degree = int(flux_degree)
        if self.flux_degree < 0:
            raise ValueError('flux_degree must be >= 0')

        self.flux_reference = float(flux_reference)
        self.flux_scale = float(flux_scale)
        if (not np.isfinite(self.flux_reference)
                or not np.isfinite(self.flux_scale)
                or self.flux_scale <= 0.0):
            raise ValueError('flux_reference must be finite and flux_scale '
                             'must be finite and > 0')

        if 'Flux' in dependencies:
            expected_shape = (self.flux_degree + 1, *self.shape)
            if flux_coeff_data is None:
                flux_coeff_data = np.zeros(expected_shape, dtype=float)
            else:
                flux_coeff_data = np.asanyarray(flux_coeff_data, dtype=float)
                if flux_coeff_data.shape != expected_shape:
                    raise ValueError('flux_coeff_data must have shape '
                                     f'{expected_shape}')
        else:
            flux_coeff_data = None
        self.flux_coeff_data = flux_coeff_data

    @property
    def has_position_dependency(self):
        return 'Position' in self.dependencies

    @property
    def has_flux_dependency(self):
        return 'Flux' in self.dependencies

    def _normalize_flux(self, flux):
        flux = np.asanyarray(flux, dtype=float)
        return (flux - self.flux_reference) / self.flux_scale

    def flux_basis_vector(self, flux):
        flux_norm = self._normalize_flux(flux)
        basis = [np.ones_like(flux_norm)]
        for power in range(1, self.flux_degree + 1):
            basis.append(flux_norm**power)
        return np.stack(basis, axis=0)

    def local_epsf_data(self, x, y, flux=None):
        if self.has_position_dependency:
            data = super().local_epsf_data(x, y)
        else:
            # With no position dependence, the baseline is the constant term.
            data = np.array(self.coeff_data[0], copy=True)

        if self.has_flux_dependency:
            if flux is None:
                # Use the reference flux when no explicit flux is supplied,
                # e.g., for generic diagnostics on the model grid.
                flux = self.flux_reference
            flux_basis = self.flux_basis_vector(flux)
            flux_correction = np.tensordot(flux_basis, self.flux_coeff_data,
                                           axes=(0, 0))
            data = data + flux_correction

        return data

    def make_image_psf(self, x, y, flux=None):
        data = self.local_epsf_data(x, y, flux=flux)
        if self.normalize_local_epsf:
            data = self._normalise_local_epsf_data(data)

        image_psf = self.epsf_class(data=data, oversampling=self.oversampling,
                                    origin=self.origin,
                                    fill_value=self.fill_value)
        trust_map = getattr(self, 'trust_map', None)
        if trust_map is not None:
            image_psf.trust_map = np.array(trust_map, copy=True)
        return image_psf


class VariableEPSFFitter(SpatialEPSFFitter):
    """
    Fit stars with a `VariableEPSFModel`.

    Parameters
    ----------
    use_time_integrated_flux : bool, optional
        If `True`, star flux is multiplied by ``star.exposure_time`` when
        evaluating flux-dependent model terms.
    """

    def __init__(self, *, fitter=None, fit_boxsize=3, progress_bar=False,
                 plot_fit_checks=False, model_weight_map=None,
                 model_weight_maxiters=1, model_weight_center_tol=1.0e-3,
                 use_time_integrated_flux=True, **fitter_kwargs):
        if fitter is None:
            fitter = TRFLSQFitter()
        super().__init__(fitter=fitter, fit_boxsize=fit_boxsize,
                         progress_bar=progress_bar,
                         plot_fit_checks=plot_fit_checks,
                         model_weight_map=model_weight_map,
                         model_weight_maxiters=model_weight_maxiters,
                         model_weight_center_tol=model_weight_center_tol,
                         **fitter_kwargs)
        self.use_time_integrated_flux = bool(use_time_integrated_flux)
        self._warned_missing_exptime = False

    def _effective_flux(self, star):
        flux = float(star.flux)
        if not self.use_time_integrated_flux:
            return flux

        exposure_time = getattr(star, 'exposure_time', None)
        if exposure_time is None:
            if not self._warned_missing_exptime:
                warnings.warn('One or more stars do not define '
                              'exposure_time; falling back to fitted flux '
                              'for flux-dependent model terms.',
                              AstropyUserWarning)
                self._warned_missing_exptime = True
            return flux

        exposure_time = float(exposure_time)
        if not np.isfinite(exposure_time) or exposure_time <= 0.0:
            if not self._warned_missing_exptime:
                warnings.warn('Encountered non-finite or non-positive '
                              'exposure_time; falling back to fitted flux '
                              'for flux-dependent model terms.',
                              AstropyUserWarning)
                self._warned_missing_exptime = True
            return flux

        return flux * exposure_time

    def __call__(self, spatial_epsf, stars):
        if len(stars) == 0:
            return stars
        if not isinstance(spatial_epsf, VariableEPSFModel):
            raise TypeError('spatial_epsf must be a VariableEPSFModel')

        fitted_stars = []
        pbar = None
        if self.progress_bar:
            pbar = add_progress_bar(total=len(stars),
                                    desc='VariableEPSFFitter')

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
                                fitted_star.center[0], fitted_star.center[1],
                                flux=self._effective_flux(fitted_star))
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
            local_epsf = spatial_epsf.make_image_psf(
                star_work.center[0], star_work.center[1],
                flux=self._effective_flux(star_work))
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


class VariableEPSFBuilder(SpatialEPSFBuilder):
    """
    Build an ePSF model with configurable dependencies on detector
    position and/or time-integrated flux.

    Notes
    -----
    This is an experimental prototype that layers a flux-dependent
    additive correction onto the spatial model.
    """

    def __init__(self, *, dependencies=('Position',), flux_degree=1,
                 flux_min_valid_samples=25,
                 use_time_integrated_flux=True, fitter=None, **kwargs):
        self.dependencies = _parse_dependencies(dependencies)
        self.flux_degree = int(flux_degree)
        if self.flux_degree < 0:
            raise ValueError('flux_degree must be >= 0')

        self.flux_min_valid_samples = int(flux_min_valid_samples)
        if self.flux_min_valid_samples <= 0:
            raise ValueError('flux_min_valid_samples must be positive')

        self.use_time_integrated_flux = bool(use_time_integrated_flux)
        self._warned_missing_exptime = False
        self._flux_reference = 1.0
        self._flux_scale = 1.0

        if fitter is None:
            fitter = SpatialEPSFFitter()
        if not isinstance(fitter, SpatialEPSFFitter):
            raise TypeError('fitter must be a SpatialEPSFFitter instance')

        if 'Position' not in self.dependencies:
            kwargs = dict(kwargs)
            kwargs['degree'] = 0

        super().__init__(fitter=fitter, **kwargs)
        self.variable_fitter = VariableEPSFFitter(
            fitter=self.fitter.fitter,
            fit_boxsize=self.fitter.fit_boxsize,
            progress_bar=self.progress_bar,
            plot_fit_checks=self.fitter.plot_fit_checks,
            model_weight_map=self.fitter.model_weight_map,
            model_weight_maxiters=self.fitter.model_weight_maxiters,
            model_weight_center_tol=self.fitter.model_weight_center_tol,
            use_time_integrated_flux=self.use_time_integrated_flux)

    def _effective_flux(self, star):
        flux = float(star.flux)
        if not self.use_time_integrated_flux:
            return flux

        exposure_time = getattr(star, 'exposure_time', None)
        if exposure_time is None:
            if not self._warned_missing_exptime:
                warnings.warn('One or more stars do not define '
                              'exposure_time; falling back to fitted flux '
                              'for flux-dependent model terms.',
                              AstropyUserWarning)
                self._warned_missing_exptime = True
            return flux

        exposure_time = float(exposure_time)
        if not np.isfinite(exposure_time) or exposure_time <= 0.0:
            if not self._warned_missing_exptime:
                warnings.warn('Encountered non-finite or non-positive '
                              'exposure_time; falling back to fitted flux '
                              'for flux-dependent model terms.',
                              AstropyUserWarning)
                self._warned_missing_exptime = True
            return flux

        return flux * exposure_time

    def _collect_effective_flux_samples(self, stars):
        fluxes = []
        for item in stars:
            if isinstance(item, LinkedEPSFStar):
                for star in item.all_good_stars:
                    fluxes.append(self._effective_flux(star))
                continue

            if item._excluded_from_fit:
                continue
            fluxes.append(self._effective_flux(item))

        return np.asarray(fluxes, dtype=float)

    @staticmethod
    def _normalize_flux_samples(fluxes):
        fluxes = np.asanyarray(fluxes, dtype=float)
        valid = np.isfinite(fluxes)
        if np.count_nonzero(valid) == 0:
            return np.zeros_like(fluxes), 1.0, 1.0

        reference = np.nanmedian(fluxes[valid])
        lo, hi = np.nanpercentile(fluxes[valid], [5.0, 95.0])
        scale = hi - lo
        if not np.isfinite(scale) or scale <= 0.0:
            scale = max(abs(reference), 1.0)
        return (fluxes - reference) / scale, reference, scale

    def _fit_flux_coefficients(self, residuals, weights, effective_flux):
        _, ny, nx = residuals.shape
        n_basis = self.flux_degree + 1
        coeff = np.zeros((n_basis, ny, nx), dtype=float)

        flux_norm, reference, scale = self._normalize_flux_samples(
            effective_flux)
        valid_flux = np.isfinite(flux_norm)

        design_full = np.vander(flux_norm, n_basis, increasing=True)

        for iy in range(ny):
            for ix in range(nx):
                z = residuals[:, iy, ix]
                w = weights[:, iy, ix]
                valid = (valid_flux & np.isfinite(z) & np.isfinite(w)
                         & (w > 0.0))
                if np.count_nonzero(valid) < self.flux_min_valid_samples:
                    continue

                z_fit = z[valid]
                design = design_full[valid]
                w_fit = np.sqrt(w[valid])

                if self._sigma_clip is not None:
                    clipped = self._sigma_clip(z_fit, axis=0, masked=True,
                                               return_bounds=False)
                    keep = ~clipped.mask
                    z_fit = clipped.data[keep]
                    design = design[keep]
                    w_fit = w_fit[keep]

                if z_fit.size < self.flux_min_valid_samples:
                    continue

                design_w = design * w_fit[:, np.newaxis]
                z_w = z_fit * w_fit
                coeffs, _, _, _ = np.linalg.lstsq(design_w, z_w, rcond=None)
                coeff[:, iy, ix] = coeffs

        return coeff, reference, scale

    def _resample_residual(self, star, spatial_model):
        local_epsf = spatial_model.make_image_psf(
            star.center[0], star.center[1], flux=self._effective_flux(star))
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

    def _select_residual_stars(self, stars, spatial_model):
        if self.residual_star_rms_clip is None:
            return copy.deepcopy(stars)

        selected = copy.deepcopy(stars)
        good_stars = selected.all_good_stars
        if len(good_stars) == 0:
            return selected

        rms_values = []
        for star in good_stars:
            local_epsf = spatial_model.make_image_psf(
                star.center[0], star.center[1],
                flux=self._effective_flux(star))
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

    def _create_initial_variable_model(self, stars, *, init_model=None):
        if init_model is not None:
            if isinstance(init_model, VariableEPSFModel):
                return init_model.deepcopy()

            if isinstance(init_model, SpatialEPSFModel):
                return VariableEPSFModel(
                    init_model.coeff_data,
                    oversampling=init_model.oversampling,
                    detector_shape=init_model.detector_shape,
                    degree=init_model.degree,
                    origin=init_model.origin,
                    fill_value=init_model.fill_value,
                    detector_origin=init_model.detector_origin,
                    detector_span=init_model.detector_span,
                    normalize_local_epsf=init_model.normalize_local_epsf,
                    trust_map=getattr(init_model, 'trust_map', None),
                    epsf_class=init_model.epsf_class,
                    dependencies=self.dependencies,
                    flux_degree=self.flux_degree,
                    flux_reference=self._flux_reference,
                    flux_scale=self._flux_scale)

            raise TypeError('init_model must be a VariableEPSFModel or '
                            'SpatialEPSFModel')

        base_model = self._create_initial_model(stars)
        return VariableEPSFModel(
            base_model.coeff_data, oversampling=self.oversampling,
            detector_shape=self.detector_shape, degree=base_model.degree,
            origin=base_model.origin, fill_value=base_model.fill_value,
            detector_origin=self.detector_origin,
            detector_span=self.detector_span,
            normalize_local_epsf=self.normalise_epsf,
            trust_map=getattr(base_model, 'trust_map', None),
            epsf_class=self.epsf_class,
            dependencies=self.dependencies,
            flux_degree=self.flux_degree,
            flux_reference=self._flux_reference,
            flux_scale=self._flux_scale)

    def build_epsf(self, stars, *, init_model=None):
        if not isinstance(stars, EPSFStars):
            raise TypeError('stars must be an EPSFStars object')

        self._models = []
        self._ppe_models = []
        self.final_ppe_model = None
        self._warned_missing_exptime = False
        self.variable_fitter._warned_missing_exptime = False
        self._resolve_detector_geometry(stars, init_model=init_model)
        self._log('VariableEPSFBuilder: creating initial variable model')
        if self.plot_diagnostics:
            self._log('VariableEPSFBuilder: plotting sample distribution')
            self.plot_sample_distribution(stars, linked_only=True)
            self._log('VariableEPSFBuilder: plotting sub-pixel distribution')
            self.plot_subpixel_distribution(stars)
        variable_model = self._create_initial_variable_model(
            stars, init_model=init_model)
        centers = stars.cutout_center_flat

        for iter_num in range(1, self.maxiters + 1):
            self._log(f'VariableEPSFBuilder: iteration {iter_num}/'
                      f'{self.maxiters} starting')
            variable_model = variable_model.deepcopy()

            self._log(f'VariableEPSFBuilder: iteration {iter_num} fitting '
                      'stars')
            fitted_stars_raw = self.variable_fitter(variable_model, stars)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} fitting '
                      'spatial PPE correction surfaces')
            spatial_ppe_model = self._fit_spatial_ppe_model(fitted_stars_raw)
            self._ppe_models.append(spatial_ppe_model)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} applying '
                      'spatial PPE corrections to fitted stars')
            fitted_stars_corrected = self._apply_spatial_ppe_corrections(
                fitted_stars_raw, spatial_ppe_model)
            self.final_ppe_model = spatial_ppe_model

            self._log(f'VariableEPSFBuilder: iteration {iter_num} applying '
                      'linked-star constraints to PPE-corrected values')
            fitted_stars = self._apply_linked_constraints(
                fitted_stars_corrected)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} selecting '
                      'stars for residual stack')
            residual_stars = self._select_residual_stars(fitted_stars,
                                                         variable_model)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} resampling '
                      'residual stack')
            residuals, weights, x_coords, y_coords, det_x, det_y, group_id = (
                self._resample_residuals(residual_stars, variable_model))

            self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                      'computing trust map from residual RMS')
            trust_map = self._compute_trust_map_from_residuals(
                residuals, weights)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} fitting '
                      'spatial residual coefficient surfaces')
            if 'Position' in self.dependencies:
                coeff_update = self._fit_residual_coefficients(
                    residuals, det_x, det_y, x_coords=x_coords,
                    y_coords=y_coords)
            else:
                coeff_update = np.zeros_like(variable_model.coeff_data)

            if self._residual_smooth_kernel is not None:
                self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                          'smoothing residual coefficient surfaces')
                for ibasis in range(coeff_update.shape[0]):
                    coeff_update[ibasis] = convolve(
                        coeff_update[ibasis], self._residual_smooth_kernel)

            if iter_num % 10 == 0 or iter_num == self.maxiters:
                self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                          'plotting diagnostics')
                self._plot_iteration_diagnostics(variable_model, coeff_update,
                                                fitted_stars_raw,
                                                fitted_stars_corrected,
                                                det_x, det_y, iter_num)
                self._plot_central_residual_fit(residuals, det_x, det_y,
                                                group_id, x_coords, y_coords,
                                                coeff_update, iter_num)
                self._plot_residual_scatter_diagnostics(
                    residuals, det_x, det_y, group_id, x_coords, y_coords,
                    iter_num)

            flux_reference = variable_model.flux_reference
            flux_scale = variable_model.flux_scale
            flux_coeff_update = None
            if 'Flux' in self.dependencies:
                self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                          'fitting flux residual coefficient surfaces')
                effective_flux = self._collect_effective_flux_samples(
                    residual_stars)
                flux_coeff_update, flux_reference, flux_scale = (
                    self._fit_flux_coefficients(residuals, weights,
                                                effective_flux))

            self._log(f'VariableEPSFBuilder: iteration {iter_num} updating '
                      'variable ePSF coefficients')
            coeff_data = variable_model.coeff_data.copy()
            coeff_data += self.residual_update_fraction * coeff_update

            flux_coeff_data = None
            if 'Flux' in self.dependencies:
                old_flux_coeff = (np.zeros_like(flux_coeff_update)
                                  if variable_model.flux_coeff_data is None
                                  else variable_model.flux_coeff_data)
                flux_coeff_data = (old_flux_coeff
                                   + self.residual_update_fraction
                                   * flux_coeff_update)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} smoothing '
                      'coefficient images')
            coeff_data = self._smooth_coefficients(coeff_data)

            sample_positions = list(zip(det_x, det_y, strict=True))

            if self.recenter_epsf:
                self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                          'recentering variable ePSF')
                coeff_data = self._recenter_coefficients(coeff_data,
                                                         sample_positions)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} '
                      'normalizing variable ePSF')
            coeff_data = self._normalise_coefficients(coeff_data,
                                                      sample_positions)

            self._log(f'VariableEPSFBuilder: iteration {iter_num} rebuilding '
                      'variable model object')
            variable_model = VariableEPSFModel(
                coeff_data, oversampling=self.oversampling,
                detector_shape=self.detector_shape,
                degree=variable_model.degree,
                origin=variable_model.origin,
                fill_value=variable_model.fill_value,
                detector_origin=self.detector_origin,
                detector_span=self.detector_span,
                normalize_local_epsf=self.normalise_epsf,
                trust_map=trust_map,
                epsf_class=self.epsf_class,
                dependencies=self.dependencies,
                flux_coeff_data=flux_coeff_data,
                flux_degree=self.flux_degree,
                flux_reference=flux_reference,
                flux_scale=flux_scale)

            self._flux_reference = flux_reference
            self._flux_scale = flux_scale
            self._models.append(variable_model)

            dx_dy = fitted_stars.cutout_center_flat - centers
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = fitted_stars.cutout_center_flat
            if (center_dist_sq.size > 0
                    and np.nanmax(center_dist_sq) < self.center_accuracy_sq):
                self._log('VariableEPSFBuilder: converged after iteration '
                          f'{iter_num}')
                stars = fitted_stars
                break

            stars = fitted_stars

        self._log('VariableEPSFBuilder: finished')
        return variable_model, stars
