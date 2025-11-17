import numpy as np
from scipy.optimize import least_squares

__all__ = ['PriorLogTRFLSQFitter']

class PriorLogTRFLSQFitter:
    """
    TRF-like fitter for photutils ImagePSF that fits log(flux) internally,
    supports Gaussian priors, and returns covariance both in internal
    (logA, x0, y0, ...) and physical (flux, x0, y0, ...) spaces.

    Usage (unchanged):
        fitter = PriorTRFLSQFitter(prior_mean=[A0,x0,y0], prior_sigma=[sA,sx,sy], ...)
        fitted_model = fitter(model=epsf, x=xx, y=yy, z=data, weights=weights)

    After fitting:
      - fitter.fit_info (OptimizeResult) contains:
          - 'param_cov_internal' : covariance in internal params (logA, x0, y0, ...)
          - 'param_uncert_internal' : sqrt(diag(param_cov_internal))
          - 'param_cov_flux' : covariance in physical params (flux, x0, y0, ...)
          - 'param_uncert_flux' : sqrt(diag(param_cov_flux))
          - 'qfit', '_fit_cond', etc.
      - The returned `model` has .flux/.x_0/.y_0 updated, and receives
        attributes model.param_cov_flux and model.param_uncert_flux where possible.
    """

    def __init__(self, prior_mean=None, prior_sigma=None,
                 rel_amp_uncert=0.3, pixel_uncert=0.5, max_nfev=2000, verbose=False):
        self.prior_mean = None if prior_mean is None else np.asarray(prior_mean, dtype=float)
        self.prior_sigma = None if prior_sigma is None else np.asarray(prior_sigma, dtype=float)
        self.rel_amp_uncert = float(rel_amp_uncert)
        self.pixel_uncert = float(pixel_uncert)
        self.max_nfev = int(max_nfev)
        self.verbose = bool(verbose)
        self.fit_info = None

    def __call__(self, model, x, y, z, weights=None, **kwargs):
        # --- prepare data arrays (flatten) ---
        data = np.asarray(z)
        px = np.asarray(x)
        py = np.asarray(y)

        if data.shape != px.shape or data.shape != py.shape:
            px = np.broadcast_to(px, data.shape)
            py = np.broadcast_to(py, data.shape)

        data_flat = data.ravel()
        px_flat = px.ravel()
        py_flat = py.ravel()
        m = data_flat.size

        # --- interpret weights -> sigma_pix ---
        if weights is None:
            sigma_pix = np.ones_like(data_flat)
        else:
            w = np.asarray(weights).ravel()
            if w.size != data_flat.size:
                w = np.broadcast_to(w, data.shape).ravel()
            # Heuristic: if any w > 1 treat as 1/sigma, else treat as sigma
            if np.any(w > 1.0):
                sigma_pix = np.clip(1.0 / w, 1e-12, None)
            else:
                sigma_pix = np.clip(w, 1e-12, None)

        # --- initial model parameters (flux, x0, y0, ...) ---
        try:
            p0_model = np.asarray(model.parameters, dtype=float).copy()
        except Exception:
            raise ValueError("model must expose .parameters (astropy/photutils API)")

        n_params_model = p0_model.size

        # Build internal initial guess: [logA, x0, y0, ...]
        flux0 = float(p0_model[0]) if n_params_model >= 1 else 1.0
        flux0_pos = max(abs(flux0), 1e-12)
        logA0 = np.log(flux0_pos)
        p0_internal = np.empty(n_params_model, dtype=float)
        p0_internal[0] = logA0
        if n_params_model > 1:
            p0_internal[1:] = p0_model[1:]

        # --- build priors in model-space, then convert to internal space ---
        if self.prior_mean is not None:
            prior_mean_model = np.asarray(self.prior_mean, dtype=float)
            if prior_mean_model.size != n_params_model:
                raise ValueError("prior_mean length must equal number of model parameters")
        else:
            prior_mean_model = p0_model.copy()

        prior_mean_internal = np.empty_like(prior_mean_model, dtype=float)
        prior_mean_internal[0] = np.log(max(abs(prior_mean_model[0]), 1e-12))
        if n_params_model > 1:
            prior_mean_internal[1:] = prior_mean_model[1:]

        if self.prior_sigma is not None:
            prior_sigma_model = np.asarray(self.prior_sigma, dtype=float)
            if prior_sigma_model.size != n_params_model:
                raise ValueError("prior_sigma length must equal number of model parameters")
            prior_sigma_internal = np.empty_like(prior_sigma_model, dtype=float)
            # flux sigma -> approximate sigma on log(flux)
            flux_sigma_model = max(1e-12, abs(prior_sigma_model[0]))
            flux_mean = max(abs(prior_mean_model[0]), flux0_pos, 1e-12)
            prior_sigma_internal[0] = flux_sigma_model / flux_mean
            if n_params_model > 1:
                prior_sigma_internal[1:] = prior_sigma_model[1:]
        else:
            prior_sigma_internal = np.empty(n_params_model, dtype=float)
            prior_sigma_internal[0] = max(1e-8, float(self.rel_amp_uncert))
            if n_params_model >= 3:
                prior_sigma_internal[1] = self.pixel_uncert
                prior_sigma_internal[2] = self.pixel_uncert
            for i in range(3, n_params_model):
                prior_sigma_internal[i] = max(1e-8, 0.1 * (abs(prior_mean_model[i]) + 1.0))

        # --- model prediction in internal parameters (logA, x0, y0) ---
        def model_prediction_internal(p_internal):
            logA = float(p_internal[0])
            A = np.exp(logA)
            if p_internal.size >= 3:
                x0 = float(p_internal[1])
                y0 = float(p_internal[2])
            elif p_internal.size == 2:
                x0 = float(p_internal[1]); y0 = 0.0
            else:
                x0 = 0.0; y0 = 0.0
            vals = model.evaluate(px_flat, py_flat, A, x0, y0)
            return np.asarray(vals).ravel()

        # --- residuals (data + prior) in internal space ---
        def residuals_internal(p_internal):
            model_vals = model_prediction_internal(p_internal)
            res_data = (model_vals - data_flat) / sigma_pix
            res_prior = (p_internal - prior_mean_internal) / prior_sigma_internal
            return np.concatenate([res_data, res_prior])

        # --- run least_squares on internal parameters ---
        x0_internal = p0_internal.copy()
        res = least_squares(residuals_internal, x0_internal, method='trf', max_nfev=self.max_nfev)

        # store OptimizeResult
        self.fit_info = res

        # split jacobian into data and prior parts for diagnostics
        J_full = res.jac                    # (m + n_params) x n_params
        m_full, n = J_full.shape
        m_data = data_flat.size
        J_data = J_full[:m_data, :]
        J_prior = J_full[m_data:, :]

        # --- update model parameters from fitted internal params ---
        fitted_internal = res.x
        fitted_logA = float(fitted_internal[0])
        fitted_flux = float(np.exp(fitted_logA))

        # update model flux and centroids if possible
        try:
            model.flux = fitted_flux
        except Exception:
            pass
        if n_params_model >= 2:
            try:
                model.x_0 = float(fitted_internal[1])
            except Exception:
                pass
        if n_params_model >= 3:
            try:
                model.y_0 = float(fitted_internal[2])
            except Exception:
                pass

        # try to update full parameters list
        try:
            p_model_fitted = np.empty_like(p0_model)
            p_model_fitted[0] = fitted_flux
            if n_params_model > 1:
                p_model_fitted[1:] = fitted_internal[1:]
            model.parameters = p_model_fitted.tolist()
        except Exception:
            try:
                model._parameters = p_model_fitted.tolist()
            except Exception:
                pass

        # --- compute covariance in internal space ---
        final_model_vals = model_prediction_internal(fitted_internal)
        resid_data = final_model_vals - data_flat
        dof = max(1, m - n)
        sigma2 = (resid_data**2).sum() / float(dof)

        JTJ = J_full.T @ J_full
        cond = np.linalg.cond(JTJ)
        try:
            cov_internal = np.linalg.inv(JTJ) * sigma2
        except np.linalg.LinAlgError:
            cov_internal = np.linalg.pinv(JTJ) * sigma2

        # store internal covariance and uncertainties
        self.fit_info['param_cov_internal'] = cov_internal
        self.fit_info['param_uncert_internal'] = np.sqrt(np.abs(np.diag(cov_internal)))
        self.fit_info['_fit_cond'] = cond

        # --- transform covariance from internal (logA, ...) to physical (flux, ...) ---
        # Build Jacobian J_trans = d(phys_params)/d(internal_params)
        # phys_params = [flux, p1, p2, ...], internal = [logA, p1, p2, ...]
        J_trans = np.eye(n)
        J_trans[0, 0] = fitted_flux   # d flux / d logA = exp(logA) = flux

        cov_flux = J_trans @ cov_internal @ J_trans.T
        param_uncert_flux = np.sqrt(np.abs(np.diag(cov_flux)))

        # attach flux-space covariance and uncertainties
        self.fit_info['param_cov_flux'] = cov_flux
        self.fit_info['param_uncert_flux'] = param_uncert_flux
        self.fit_info['fitted_flux'] = fitted_flux

        # compute qfit (photutils definition: sum|res| / fitted_flux)
        qfit = np.sum(np.abs(data_flat - final_model_vals)) / max(1e-12, fitted_flux)
        self.fit_info['qfit'] = qfit

        # attach to model if possible
        try:
            model.param_cov_flux = cov_flux
            model.param_uncert_flux = param_uncert_flux
            model.qfit = qfit
        except Exception:
            pass

        # optional verbose diagnostics prints
        if self.verbose:
            print("||J_data||_fro", np.linalg.norm(J_data))
            print("||J_prior||_fro", np.linalg.norm(J_prior))
            print("singular values (data-only):", np.linalg.svd(J_data, compute_uv=False))
            print("singular values (full):", np.linalg.svd(J_full, compute_uv=False))
            approx_prior = 1.0 / np.array([max(1e-12, prior_sigma_internal[0])] + \
                                          [max(1e-12, prior_sigma_internal[i]) for i in range(1, n)])
            print("approx prior row magnitudes (1/prior_sigma_internal):", approx_prior)
            print("cond(J^T J) =", cond)
            print("param_uncert_flux:", param_uncert_flux)
            print("qfit:", qfit)

        return model
