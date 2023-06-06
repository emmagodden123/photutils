# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import inspect
import warnings

import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import overlap_slices
from astropy.table import QTable, Table, hstack
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.utils._optional_deps import HAS_TQDM
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['PSFPhotometry']


class PSFPhotometry:
    """
    Class to perform PSF photometry.

    This class implements a flexible PSF photometry algorithm that can
    find sources in an image, group overlapping sources, fit the PSF
    model to the sources, and subtract the fit PSF models from the
    image.
    """

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=LevMarLSQFitter(), maxiters=100, aperture_radius=None,
                 progress_bar=None):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_callable(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.maxiters = self._validate_maxiters(maxiters)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self.finder_results = []
        self.fit_error_indices = []
        self._fit_group_models = []
        self.fit_models = []
        self.fit_infos = []
        self._fit_param_err = []
        self._fit_nsources = []
        self._ungroup_indices = []

    @staticmethod
    def _validate_model(psf_model):
        if not isinstance(psf_model, Fittable2DModel):
            raise TypeError('psf_model must be an astropy Fittable2DModel')
        return psf_model

    @staticmethod
    def _validate_callable(obj, name):
        if obj is not None and not callable(obj):
            raise TypeError(f'{name!r} must be a callable object')
        return obj

    def _validate_maxiters(self, maxiters):
        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            warnings.warn('"maxiters" will be ignored because it is not '
                          'accepted by the input fitter __call__ method',
                          AstropyUserWarning)
            maxiters = None
        return maxiters

    @staticmethod
    def _validate_radius(radius):
        if radius is not None and (not np.isscalar(radius)
                                   or radius <= 0 or ~np.isfinite(radius)):
            raise ValueError('radius must be a strictly-positive scalar')
        return radius

    def _validate_array(self, array, name, data_shape=None):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                raise ValueError(f'{name} must be a 2D array.')
            if data_shape is not None and array.shape != data_shape:
                raise ValueError(f'data and {name} must have the same shape.')
        return array

    @staticmethod
    def _validate_params(init_params):
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            raise TypeError('init_params must be an astropy Table')

        columns = ('x_init', 'y_init')
        for column in columns:
            if column not in init_params.columns:
                raise ValueError(f'{column!r} must be a column in '
                                 'init_params')

        return init_params.copy()

    @staticmethod
    def _make_mask(image, mask):
        def warn_nonfinite():
            warnings.warn('Input data contains unmasked non-finite values '
                          '(NaN or inf), which were automatically ignored.',
                          AstropyUserWarning)

        # if NaNs are in the data, no actual fitting takes place
        # https://github.com/astropy/astropy/pull/12811
        finite_mask = ~np.isfinite(image)

        if mask is not None:
            mask |= finite_mask
            if np.any(finite_mask & ~mask):
                warn_nonfinite()
        else:
            mask = finite_mask
            if np.any(finite_mask):
                warn_nonfinite()

        return mask

    def _add_progress_bar(self, iterable, desc=None):
        if self.progress_bar and HAS_TQDM:
            try:
                from ipywidgets import FloatProgress  # noqa: F401
                from tqdm.auto import tqdm
            except ImportError:
                from tqdm import tqdm

            iterable = tqdm(iterable, desc=desc)  # pragma: no cover

        return iterable

    def _get_aper_fluxes(self, data, mask, init_params):
        # TODO: flexible input column names
        xpos = init_params['x_init']
        ypos = init_params['y_init']
        apertures = CircularAperture(zip(xpos, ypos), r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)
        return flux

    def _make_init_params(self, data, mask, sources):
        """
        sources : `~astropy.table.Table`
            Output from star finder with 'xcentroid' and 'ycentroid'
            columns'.
        """
        init_params = QTable()
        init_params['id'] = np.arange(len(sources)) + 1
        # TODO: flexible finder column names
        init_params['x_init'] = sources['xcentroid']
        init_params['y_init'] = sources['ycentroid']
        init_params['flux_init'] = self._get_aper_fluxes(data, mask,
                                                         init_params)

        return init_params

    def _param_map(self):
        # TODO: generalize this mapping based of self.psf_model
        param_map = {}
        param_map['x_init'] = 'x_0'
        param_map['y_init'] = 'y_0'
        param_map['flux_init'] = 'flux'

        fit_param_map = {val: key.replace('_init', '_fit')
                         for key, val in param_map.items()}

        return param_map, fit_param_map

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources within
        a group.
        """
        param_map = self._param_map()[0]

        for index, source in enumerate(sources):
            model = self.psf_model.copy()
            for param, model_param in param_map.items():
                setattr(model, model_param, source[param])
                model.name = source['id']

            if index == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def _define_fit_coords(self, sources, mask):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf

        hshape = (self.fit_shape - 1) // 2
        yi = []
        xi = []
        for row in sources:
            # bbox "slice indices" (max is non-inclusive)
            xcen = int(row['x_init'] + 0.5)
            ycen = int(row['y_init'] + 0.5)
            xmin = min((xmin, xcen - hshape[1]))
            xmax = max((xmax, xcen + hshape[1] + 1))
            ymin = min((ymin, ycen - hshape[0]))
            ymax = max((ymax, ycen + hshape[0] + 1))
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            xi.append(xx)
            yi.append(yy)

        xi = np.array(xi).ravel()
        yi = np.array(yi).ravel()
        # find unique (x, y) pairs
        yi, xi = np.unique(np.column_stack((yi, xi)), axis=0).T
        # yi, xi = np.array(list(set(zip(yi, xi)))).T

        if mask is not None:
            inv_mask = ~mask[yi, xi]
            yi = yi[inv_mask]
            xi = xi[inv_mask]

        return yi, xi

    @staticmethod
    def _split_compound_model(model, chunk_size):
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i: i + chunk_size]

    def _split_groups(self, models, infos):
        psf_nsub = self.psf_model.n_submodels

        fit_models = []
        fit_infos = []
        for model, info in zip(models, infos):
            model_nsub = model.n_submodels

            # model is for a single source (which may be compound)
            if model_nsub == psf_nsub:
                fit_models.append(model)
                fit_infos.append(info)
                self._fit_nsources.append(1)
                continue

            # model is a grouped model for multiple sources
            fit_models.extend(self._split_compound_model(model, psf_nsub))
            nsources = model_nsub // psf_nsub
            self._fit_nsources.extend([nsources] * nsources)
            fit_infos.extend([info] * nsources)  # views

        if len(fit_models) != len(fit_infos):
            raise ValueError('fit_models and fit_infos have different lengths')

        self._fit_nsources = np.array(self._fit_nsources, dtype=int)
        self._fit_nsources = self._fit_nsources[self._ungroup_indices]
        z1 = [fit_models[i] for i in self._ungroup_indices]
        z2 = [fit_infos[i] for i in self._ungroup_indices]

        # now change the sorting from group_id to source id order
        # the model.name attribute stores the source index
        fit_models, fit_infos = zip(*sorted(zip(fit_models, fit_infos),
                                            key=lambda pair: pair[0].name))

        assert z1 == list(fit_models)
        assert z2 == list(fit_infos)

        return list(fit_models), list(fit_infos)

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        if self.maxiters is not None:
            kwargs = {'maxiter': self.maxiters}
        else:
            kwargs = {}

        sources = init_params.group_by('group_id')
        self._ungroup_indices = np.argsort(sources['id'].value)
        sources = sources.groups
        sources = self._add_progress_bar(sources, desc='Fit source/group')

        fit_models = []
        fit_infos = []
        for sources_ in sources:  # fit in group_id order
            psf_model = self._make_psf_model(sources_)
            yi, xi = self._define_fit_coords(sources_, data.shape, mask)
            cutout = data[yi, xi]

            if cutout.shape == (0,):
                fit_model = psf_model  # no fit was performed (initial model)
                fit_info = {'message': 'Source was not fit because it was '
                            'completely masked'}
            else:
                if error is not None:
                    weights = 1.0 / error[yi, xi]
                else:
                    weights = None

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyUserWarning)
                    fit_model = self.fitter(psf_model, xi, yi, cutout,
                                            weights=weights, **kwargs)
                    fit_info = self.fitter.fit_info.copy()

            fit_models.append(fit_model)
            fit_infos.append(fit_info)

        # split the groups and return objects in source-id order
        self._fit_group_models = fit_models
        fit_models, fit_infos = self._split_groups(fit_models, fit_infos)
        self.fit_models = fit_models
        self.fit_infos = fit_infos

        return fit_models

    def _set_fit_info_attrs(self):
        for index, fit_info in enumerate(self.fit_infos):
            ierr = fit_info.get('ierr', None)
            if ierr not in (1, 2, 3, 4):  # all good flags defined by scipy
                self.fit_error_indices.append(index)

            param_cov = fit_info.get('param_cov', None)
            nparam = 3
            if param_cov is None:
                param_err = np.array([np.nan] * nparam)
            else:
                param_err = np.sqrt(np.diag(param_cov))

            if self._fit_nsources[index] == 1:
                self._fit_param_err.append(param_err)
            else:
                param_errs = []
                for i in range(0, len(param_err),  nparam):
                    param_errs.append(param_err[i:i + nparam])
                self._fit_param_err.extend(param_errs)

        self._fit_param_err = np.array(self._fit_param_err)
        self.fit_error_indices = np.array(self.fit_error_indices, dtype=int)

    def _model_params_to_table(self, models):
        param_map = self._param_map()[1]

        params = []
        for model in models:
            mparams = []
            for model_param in param_map.keys():
                mparams.append(getattr(model, model_param).value)
            params.append(mparams)
        vals = np.transpose(params)

        colnames = param_map.values()
        table = QTable()
        for index, colname in enumerate(colnames):
            table[colname] = vals[index]

        return table

    def _define_flags(self):
        flags = np.zeros(len(self.fit_infos), dtype=int)
        flags[self.fit_error_indices] = 1

        idx = []
        for index, fit_info in enumerate(self.fit_infos):
            if 'completely masked' in fit_info['message']:
                idx.append(index)
        flags[idx] = 2

        return flags

    def __call__(self, data, *, mask=None, error=None, init_params=None):
        """
        Perform PSF photometry.
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        data = self._validate_array(data, 'data')
        mask = self._make_mask(data,
                               self._validate_array(mask, 'mask',
                                                    data_shape=data.shape))
        init_params = self._validate_params(init_params)  # also copies

        if (self.aperture_radius is None
            and (init_params is None
                 or 'flux_init' not in init_params.colnames)):
            raise ValueError('aperture_radius must be defined if init_params '
                             'is not input or if a "flux_init" column is '
                             'not in init_params')

        if init_params is None:
            if self.finder is None:
                raise ValueError('finder must be defined if init_params '
                                 'is not input')

            sources = self.finder(data, mask=mask)
            self.finder_results.append(sources)
            if sources is None:
                return None

            init_params = self._make_init_params(data, mask, sources)
        else:
            colnames = init_params.colnames
            if 'id' not in colnames:
                init_params['id'] = np.arange(len(init_params)) + 1

            if 'flux_init' not in colnames:
                init_params['flux_init'] = self._get_aper_fluxes(data, mask,
                                                                 init_params)

            if 'group_id' in colnames:
                # grouper is ignored if group_id is input in init_params
                self.grouper = None

        if self.grouper is not None:
            # TODO: change grouper API
            # init_params['group_id'] = self.grouper(init_params)
            init_params = self.grouper(init_params)

        # no grouping
        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        # order init_params columns
        colnames = ('id', 'group_id', 'x_init', 'y_init', 'flux_init')
        init_params = init_params[colnames]

        fit_models = self._fit_sources(data, init_params, error=error,
                                       mask=mask)

        self._set_fit_info_attrs()

        # create output table
        fit_sources = self._model_params_to_table(fit_models)  # ungrouped
        if len(init_params) != len(fit_sources):
            raise ValueError('init_params and fit_sources tables have '
                             'different lengths')
        source_tbl = hstack((init_params, fit_sources))

        source_tbl['flags'] = self._define_flags()

        if len(self.fit_error_indices) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table, and '
                          'the "fit_error_indices" and "fit_infos" attributes '
                          'for more information.', AstropyUserWarning)

        return source_tbl

    def _get_psf_param_names(self):
        """
        Get the names of the PSF model parameters corresponding to x, y,
        and flux.

        The PSF model must either define 'xname', 'yname', and
        'fluxname' attributes or have parameters called 'x_0', 'y_0',
        and 'flux'. Otherwise, a `ValueError` is raised.
        """
        keys = [('xname', 'x_0'), ('yname', 'y_0'), ('fluxname', 'flux')]
        names = []
        for key in keys:
            try:
                name = getattr(self.psf_model, key[0])
            except AttributeError as exc:
                if key[1] in self.psf_model.param_names:
                    name = key[1]
                else:
                    msg = 'Could not find PSF parameter names'
                    raise ValueError(msg) from exc

            names.append(name)

        return tuple(names)

    def make_model_image(self, shape, psf_shape):
        fit_models = self.fit_models

        data = np.zeros(shape)
        xname, yname, _ = self._get_psf_param_names()

        desc = 'Model image'
        fit_models = self._add_progress_bar(fit_models, desc=desc)

        # fit_models must be a list of individual, not grouped, PSF
        # models, i.e., there should be one PSF model (which may be
        # compound) for each source
        for fit_model in fit_models:
            x0 = getattr(fit_model, xname).value
            y0 = getattr(fit_model, yname).value
            slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0), mode='trim')
            yy, xx = np.mgrid[slc_lg]
            data[slc_lg] += fit_model(xx, yy)

        return data

    def make_residual_image(self, data, psf_shape):
        return data - self.make_model_image(data.shape, psf_shape)
