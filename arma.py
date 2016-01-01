import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize

#max number of steps going back in prediction for noise estimate
LIMIT = 50


#class for handling the ugly side (sample (p)acf, model fitting, model evaluation...)
class ARMA:
    def __init__(self, data, d=0, subtract_mean=True, box_cox=None):
        #transform and save data
        self._mean = np.mean(data)
        mean = 0
        if subtract_mean:
            mean = self._mean
        self.set_transformation(Transform(d=d, mean=mean, box_cox=box_cox))
        self._data = self._transform(data)

        self.model = None

        #dict for caching
        self._sample_auto_covariance = {}

        #lists of implemented fitting methods
        self._implemented_ar_methods = ('durbin_levinson',
                                        'yule_walker',
                                        'min_reduced_likelihood',
                                        'least_squares',
                                        'kalman')
        self._implemented_ma_methods = ('durbin_levinson',
                                        'min_reduced_likelihood',
                                        'least_squares',
                                        'kalman')
        self._implemented_arma_methods = ('durbin_levinson',
                                          'min_reduced_likelihood',
                                          'least_squares',
                                          'kalman')
        self._implemented_prediction_method = ('innovations_algo',
                                               'kalman')
        self._implemented_likelihood_methods = ('innovations_algo',
                                                'kalman')

    def set_transformation(self, transformation):
        self._transformation = transformation
        self._transform = transformation.transform
        self._backtransform = transformation.backtransform

    def sample_autocovariance(self, lag):
        lag = abs(lag)
        if lag in self._sample_auto_covariance:
            return self._sample_auto_covariance[lag]
        if lag >= len(self._data):
            raise ValueError('lag out of range')
        sample_autocov = self._compute_sample_autocovariance(lag)
        self._sample_auto_covariance[lag] = sample_autocov
        return sample_autocov

    def _compute_sample_autocovariance(self, lag):
        n = len(self._data)
        if self._transformation.is_mean_corrected():
            mean = 0
        else:
            mean = self._mean
        return np.sum((self._data[-(n - lag):] - mean) * (self._data[:n - lag] - mean)) / n

    def sample_acf(self, lag):
        return self.sample_autocovariance(lag) / self.sample_autocovariance(0)

    def sample_covariance_matrix(self, k):
        cov_matrix = np.matrix(np.zeros([k, k]), copy=False)
        for i in range(k):
            for j in range(k):
                cov_matrix[i, j] = self.sample_autocovariance(i - j)
        return cov_matrix

    def sample_pacf(self, k):
        if k == 0:
            return 1
        Gamma_k = self.sample_covariance_matrix(k)
        gamma_k = np.matrix([self.sample_autocovariance(l) for l in range(1, k + 1)]).T
        return (np.linalg.inv(Gamma_k) * gamma_k).item((k - 1, 0))

    def plot_ACF(self, limit=None):
        if not limit:
            limit = len(self._data)
        AC = np.zeros(limit - 1)
        for lag in range(1, limit):
            AC[lag - 1] = self.sample_acf(lag)
        plt.bar(list(range(1, limit)), AC)
        plt.axhline(1.96 / np.sqrt(len(self._data)), linestyle='--', alpha=0.6)
        plt.axhline(-1.96 / np.sqrt(len(self._data)), linestyle='--', alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('ACF')

    def plot_PACF(self, limit=None):
        if not limit:
            limit = len(self._data)
        PAC = np.zeros(limit - 1)
        for lag in range(1, limit):
            PAC[lag - 1] = self.sample_pacf(lag)
        plt.bar(list(range(1, limit)), PAC)
        plt.axhline(1.96 / np.sqrt(len(self._data)), linestyle='--', alpha=0.6)
        plt.axhline(-1.96 / np.sqrt(len(self._data)), linestyle='--', alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('PACF')

    def fit_ar(self, p, method='durbin_levinson'):
        if method not in self._implemented_ar_methods:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_ar_methods))

        if p == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            self.model = self._fit_ar_durbin_levinson(p)
        elif method == 'yule_walker':
            self.model = self._fit_ar_yule_walker(p)
        elif method == 'min_reduced_likelihood':
            self.model = self._fit_arma_max_likelihood(p=p)
        elif method == 'least_squares':
            self.model = self._fit_arma_least_squares(p=p)
        elif method == 'kalman':
            self.model = self._fit_arma_max_likelihood_kalman(p=p)

    def _fit_ar_yule_walker(self, p):
        Gamma = self.sample_covariance_matrix(p)
        gamma = np.matrix([self.sample_autocovariance(l) for l in range(1, p + 1)]).T
        coefs = (np.linalg.inv(Gamma) * gamma).getA1()
        sigma = (self.sample_autocovariance(0) - coefs * gamma)[0, 0]
        return PureARMA(phi=coefs, sigma_sq=sigma)

    def _fit_ar_durbin_levinson(self, p):
        nu = np.zeros(p)
        phi = [np.zeros(m + 1) for m in range(p)]
        nu[0] = self.sample_autocovariance(0) * (1 - self.sample_acf(1) ** 2)
        phi[0][0] = self.sample_acf(1)
        for m in range(1, p):
            phi[m][m] = (
                self.sample_autocovariance(m + 1) -
                sum(phi[m - 1][j] * self.sample_autocovariance(m - j) for j in range(m))
            ) / nu[m - 1]
            for j in range(m):
                phi[m][j] = phi[m - 1][j] - phi[m][m] * phi[m - 1][m - 1 - j]
            nu[m] = nu[m - 1] * (1 - phi[m][m] ** 2)
        return PureARMA(phi=phi[p - 1], sigma_sq=nu[p - 1])

    #ToDo: Write tests
    def get_confidence_interval_ar_d_l(self):
        return [1.96 *
                np.sqrt((self.model.get_sigma_sq() *
                np.linalg.inv(self.sample_covariance_matrix(self.model.get_ar_order())))[j, j]) /
                np.sqrt(len(self._data))
                for j in range(self.model.get_ar_order())]

    def fit_ma(self, q, method='durbin_levinson'):
        if method not in self._implemented_ma_methods:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_ma_methods))

        if q == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            self.model = self._fit_ma_durbin_levinson(q)
        elif method == 'min_reduced_likelihood':
            self.model = self._fit_arma_max_likelihood(q=q)
        elif method == 'least_squares':
            self.model = self._fit_arma_least_squares(q=q)
        elif method == 'kalman':
            self.model = self._fit_arma_max_likelihood_kalman(q=q)

    def _fit_ma_durbin_levinson(self, q):
        nu = np.zeros(q + 1)
        theta = [np.zeros(m + 1) for m in range(q)]
        nu[0] = self.sample_autocovariance(0)
        for m in range(q):
            for k in range(m + 1):
                theta[m][m - k] = (
                    self.sample_autocovariance(m - k + 1) -
                    sum(theta[m][m - j] * theta[k - 1][k - j - 1] * nu[j] for j in range(k))
                ) / nu[k]
            nu[m + 1] = self.sample_autocovariance(0) -\
                sum(theta[m][m - j] ** 2 * nu[j] for j in range(m + 1))
        return PureARMA(theta=theta[q - 1], sigma_sq=nu[q])

    #ToDo write tests
    def get_confidence_interval_ma_d_l(self):
        return [1.96 *
                np.sqrt(sum(self.model.get_theta(k) ** 2 for k in range(j + 1))) /
                np.sqrt(len(self._data))
                for j in range(self.model.get_ma_order())]

    def fit_arma(self, p, q, method='durbin_levinson', **kwargs):
        if method not in self._implemented_arma_methods:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_arma_methods))

        if q == 0 and p == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            if 'm' in kwargs:
                m = kwargs['m']
            else:
                m = p + q
            self.model = self._fit_arma_durbin_levinson(p, q, m=m)
        elif method == 'min_reduced_likelihood':
            self.model = self._fit_arma_max_likelihood(p=p, q=q)
        elif method == 'least_squares':
            self.model = self._fit_arma_least_squares(p=p, q=q)
        elif method == 'kalman':
            self.model = self._fit_arma_max_likelihood_kalman(p=p, q=q)

    #ToDo test this
    def _fit_arma_durbin_levinson(self, p, q, m=0):
        m = max(m, p + q)
        ma_model = self._fit_ma_durbin_levinson(m)
        estimation_matrix = np.matrix([[ma_model.get_theta(q + j - i)
                                        for i in range(p)] for j in range(p)])
        estimation_matrix = np.linalg.inv(estimation_matrix)
        phi = estimation_matrix * np.matrix([ma_model.get_theta(q + i + 1) for i in range(p)]).T
        phi = np.array(phi).flatten()
        theta = np.zeros(q)
        for j in range(1, q + 1):
            theta_j = ma_model.get_theta(j)
            for i in range(1, min(j, p) + 1):
                if i == j:
                    theta_j -= phi[j - 1]
                else:
                    theta_j -= phi[i - 1] * a_model.get_theta(j - i)
            theta[j - 1] = theta_j
        return PureARMA(phi, theta, ma_model.get_sigma_sq())

    def get_training_predictions(self, model=None, method='kalman'):
        return self.get_one_step_predictors(len(self._data), model, method=method)

    def get_one_step_predictor(self, n, model=None, method='kalman'):
        return self.get_one_step_predictors(n, model, method=method)[n]

    def get_one_step_predictors(self, n, model=None, method='kalman'):
        if method not in self._implemented_prediction_method:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_prediction_method))
        if n > len(self._data):
            raise ValueError('One step prediction only possible for one step ahead')
        if n == 0:
            return np.matrix([[0]])
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        if method == 'innovations_algo':
            return self._classic_one_step_prediction(n, model)
        elif method == 'kalman':
            return self._kalman_predicitons(n, model)

    #ToDo refactor using numpy arrays
    #Brockwell, Davis p. 175, ch. 5.3
    def _classic_one_step_prediction(self, n, model):
        m = max(model.get_ar_order(), model.get_ma_order())
        predictions = np.zeros(max(n, m) + 1)
        for k in range(1, m):
            predictions[k] = sum(
                model.get_innovation_coef(k, j) * (self._data[k - j] - predictions[k - j])
                for j in range(1, k + 1)
            )
        for k in range(m, n + 1):
            k_limited = min(LIMIT + m, k)
            predictions[k] = sum(
                model.get_phi(j) * self._data[k - j] for j in range(1, model.get_ar_order() + 1)
            ) + sum(
                model.get_innovation_coef(k_limited, j) * (self._data[k - j] - predictions[k - j])
                for j in range(1, model.get_ma_order() + 1)
            )
        return predictions

    def _kalman_predicitons(self, n, model):
        r = max(model.get_ar_order(), model.get_ma_order() + 1)
        state_model = model.get_state_space_repr()
        predictions = np.zeros(n + 1)
        last_pred_state = np.matrix(np.zeros([r, 1]))
        for t in range(1, n + 1):
            last_pred_state = state_model.get_F() * last_pred_state +\
                state_model.get_pred_theta(t) * np.linalg.pinv(state_model.get_pred_delta(t)) *\
                (self._data[t - 1] - predictions[t - 1])
            predictions[t] = state_model.get_G() * last_pred_state
        return predictions

    def get_weighted_sum_squared_residuals(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        predictions = self.get_training_predictions(model=model, method='innovations_algo')
        rs = model.get_first_rs(len(self._data))
        return np.sum((self._data - predictions[:-1]) ** 2 / rs)

    def _wsum_residuals_by_param(self, params, p, q):
        phi = params[:p]
        theta = params[p:p + q]
        sigma_sq = 1
        model = PureARMA(phi=phi, theta=theta, sigma_sq=sigma_sq)
        return self.get_weighted_sum_squared_residuals(model=model)

    def get_likelihood(self, model=None, method='innovations_algo'):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        if method not in self._implemented_likelihood_methods:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_likelihood_methods))
        if method == 'innovations_algo':
            return self._likelihood_innovations(model)
        elif method == 'kalman':
            return self._likelihood_kalman(model)

    def _likelihood_innovations(self, model):
        rs = model.get_first_rs(len(self._data))
        return np.prod(np.divide(np.exp(self._likelihood_exponent(rs, model)),
                       self._likelihood_dividend(rs, model)))

    def _likelihood_kalman(self, model):
        predictions = (self.get_training_predictions(model=model, method='kalman'))[:-1]
        residuals = predictions - self._data
        errors = np.zeros(len(self._data))
        for t in range(len(self._data)):
            errors[t] = model.get_state_space_repr().get_pred_delta(t + 1)
        normalizing_dividend = np.sqrt(2 * np.pi) * np.sqrt(errors)
        return np.prod(np.exp(-0.5 * residuals ** 2 / errors) / normalizing_dividend)

    def _likelihood_exponent(self, rs, model):
        predictions = self.get_training_predictions(model=model, method='innovations_algo')
        return np.divide((self._data - predictions[:-1]) ** 2, (-2 * rs * model.get_sigma_sq()))

    def _likelihood_dividend(self, rs, model):
        return np.sqrt(rs * (2 * np.pi * model.get_sigma_sq()))

    def get_loglikelihood(self, model=None, method='innovations_algo'):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        if method not in self._implemented_likelihood_methods:
            raise ValueError('unknown method, implemented methods:' +
                             str(self._implemented_likelihood_methods))
        if method == 'innovations_algo':
            return self._loglikelihood_innovations(model)
        if method == 'kalman':
            return self._loglikelihood_kalman(model)

    def _loglikelihood_innovations(self, model):
        rs = model.get_first_rs(len(self._data))
        return np.sum(self._likelihood_exponent(rs, model) -
                      np.log(self._likelihood_dividend(rs, model)))

    def _loglikelihood_kalman(self, model):
        predictions = (self.get_training_predictions(model=model, method='kalman'))[:-1]
        residuals = predictions - self._data
        errors = np.zeros(len(self._data))
        for t in range(len(self._data)):
            errors[t] = model.get_state_space_repr().get_pred_delta(t + 1)
        normalizing_dividend = np.sqrt(2 * np.pi) * np.sqrt(errors)
        return np.sum((-0.5 * residuals ** 2 / errors) - np.log(normalizing_dividend))

    def get_reduced_likelihood(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        rs = model.get_first_rs(len(self._data))
        return np.log(self.get_weighted_sum_squared_residuals(model=model) / len(self._data)) +\
            np.sum(np.log(rs)) / len(self._data)

    def _reduced_likelihood_by_param(self, params, p, q):
        phi = params[:p]
        theta = params[p:p + q]
        sigma_sq = 1
        model = PureARMA(phi=phi, theta=theta, sigma_sq=sigma_sq)
        return self.get_reduced_likelihood(model=model)

    def _kalman_likelihood_by_param(self, params, p, q):
        phi = params[:p]
        theta = params[p:p + q]
        sigma_sq = params[-1]
        model = PureARMA(phi=phi, theta=theta, sigma_sq=sigma_sq)
        return self.get_likelihood(model=model, method='kalman')

    #ToDo testing and error handling if minimizations fails
    def _fit_arma_max_likelihood_kalman(self, p=0, q=0):
        start_params = self._calculate_initial_coeffs(p, q)

        def to_minimize(params):
                    return -self._kalman_likelihood_by_param(params, p, q)
        opt_params = scipy.optimize.minimize(to_minimize, x0=start_params)['x']

        opt_phi = opt_params[:p]
        opt_theta = opt_params[p:p + q]
        opt_sigma_sq = opt_params[-1]
        return PureARMA(opt_phi, opt_theta, opt_sigma_sq)

    #ToDo testing and error handling if minimizations fails
    def _fit_arma_max_likelihood(self, p=0, q=0):
        start_params = self._calculate_initial_coeffs(p, q)[:-1]

        def to_minimize(params):
                    return self._reduced_likelihood_by_param(params, p, q)
        opt_params = scipy.optimize.minimize(to_minimize, x0=start_params)['x']

        opt_phi = opt_params[:p]
        opt_theta = opt_params[p:p + q]
        opt_sigma_sq = self._wsum_residuals_by_param(opt_params, p, q) / len(self._data)
        return PureARMA(opt_phi, opt_theta, opt_sigma_sq)

    #ToDo testing and error handling if minimizations fails
    def _fit_arma_least_squares(self, p=0, q=0):
        start_params = self._calculate_initial_coeffs(p, q)[:-1]

        def to_minimize(params):
                    return self._wsum_residuals_by_param(params, p, q)
        opt_params = scipy.optimize.minimize(to_minimize, x0=start_params)['x']

        opt_phi = opt_params[:p]
        opt_theta = opt_params[p:p + q]
        opt_sigma_sq = self._wsum_residuals_by_param(opt_params, p, q) / len(self._data)
        return PureARMA(opt_phi, opt_theta, opt_sigma_sq)

    def _calculate_initial_coeffs(self, p, q):
        if p == 0:
            start_model = self._fit_ma_durbin_levinson(q)
        elif q == 0:
            start_model = self._fit_ar_durbin_levinson(p)
        else:
            start_model = self._fit_arma_durbin_levinson(p, q)
        return np.hstack(start_model.get_params())

    #Brockwell, Davis p. 287
    def get_aicc(self, model=None, method='kalman'):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        return -2 * self.get_loglikelihood(model=model, method=method) +\
            2 * (model.get_ar_order() + model.get_ma_order() + 1) * len(self._data) /\
            (len(self._data) - model.get_ar_order() - model.get_ma_order() - 2)

    #Brockwell, Davis p. 304
    def get_aic(self, model=None, method='kalman'):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        return -2 * self.get_loglikelihood(model=model, method=method) +\
            2 * (model.get_ar_order() + model.get_ma_order() + 1)

    #Brockwell, Davis p. 302
    def get_fpe(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        if model.get_ma_order() > 0:
            raise ValueError('FPE only defined for AR models!')
        return model.get_sigma_sq() *\
            (len(self._data) + model.get_ar_order()) /\
            (len(self._data) - model.get_ar_order())

    #ToDo: test
    #Brockwell, Davis p. 304
    def get_bic(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        n = len(self._data)
        p = model.get_ar_order()
        q = model.get_ma_order()
        sq = model.get_sigma_sq()

        return (n - p - q) * np.log(n * sq / (n - p - q)) +\
            n * (1 + np.log(np.sqrt(2 * np.pi))) +\
            (p + q) * np.log((np.sum(self._data ** 2) - n * sq) / (p + q))

    def get_residuals(self, model=None, method='kalman'):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        predictions = self.get_training_predictions(model=model, method=method)[:-1]
        residuals = self._data - predictions
        return residuals

    #Brockwell, Davis p. 313
    def difference_sign_test(self, model=None, residuals=None):
        if residuals is None:
            if model is None:
                if self.model is None:
                    raise ValueError('no model specified')
                else:
                    model = self.model
            residuals = self.get_residuals(model=model, method='kalman')
        n = len(self._data)
        n_sign_diffs = self._differene_sign(residuals)
        expected_sign_diffs = (n - 1) / 2
        variance = (n + 1) / 12
        return abs(n_sign_diffs - expected_sign_diffs) / variance

    def _differene_sign(self, ar):
        increases = ar[1:] > ar[:-1]
        return np.sum(increases)

    #Brockwell, Davis p. 312
    def turning_point_test(self, model=None, residuals=None):
        if residuals is None:
            if model is None:
                if self.model is None:
                    raise ValueError('no model specified')
                else:
                    model = self.model
            residuals = self.get_residuals(model=model, method='kalman')
        n = len(self._data)
        n_turning_pts = self._turning_points(residuals)
        expected_tpts = 2 * (n - 2) / 3
        variance = (16 * n - 29) / 90
        return abs(n_turning_pts - expected_tpts) / variance

    def _turning_points(self, ar):
        increases = ar[1:] > ar[:-1]
        tp = (increases[1:] != increases[:-1])
        return np.sum(tp)

    def fit_summary(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        residuals = self.get_residuals(model=model, method='kalman')
        print('mean residuals: {}'.format(round(np.mean(residuals), 3)))
        print('aicc: {}'.format(self.get_aicc(model=model, method='kalman')))
        print('aic: {}'.format(self.get_aic(model=model, method='kalman')))
        print('bic: {}'.format(self.get_bic(model=model)))
        print('turning point test: {}'.format(
            round(self.turning_point_test(residuals=residuals), 3)))
        print('sign difference test: {}'.format(
            round(self.difference_sign_test(residuals=residuals), 3)))
        plt.plot(residuals)


#class for handling the pure math side, not supposed to see data
class PureARMA:
    def __init__(self, phi=None, theta=None, sigma_sq=1):
        if phi is None:
            phi = []
        if theta is None:
            theta = []
        self._phi = np.array(phi)
        self._theta = np.array(theta)
        self._sigma_sq = sigma_sq

        self._p = len(phi)
        self._q = len(theta)
        self._m = max(self._p, self._q)

        self._state_space_representation = None

        #dicts for caching
        self._ma_infty_coefs = {}
        self._acf = {}
        self._innovation_coefs = {}
        self._innovation_errors = {}

    def get_params(self):
        return (self._phi, self._theta, self._sigma_sq)

    def get_sigma_sq(self):
        return self._sigma_sq

    def get_phi(self, k):
        if k == 0:
            return 1
        if k <= self._p:
            return self._phi[k - 1]
        return 0

    def get_ar_order(self):
        return self._p

    def get_ma_order(self):
        return self._q

    def _get_ar_coeff(self, k):
        if k == 0:
            return 1
        if k < 0:
            return 0
        return -self.get_phi(k)

    def get_theta(self, k):
        if k == 0:
            return 1
        if k <= self._q:
            return self._theta[k - 1]
        return 0

    #Brockwell, Davis p. 468 ex.12.1.5
    def get_state_space_repr(self):
        if self._state_space_representation is None:
            F = self._compute_F()
            G = self._compute_G()
            Q = self._compute_Q()
            r = max(self.get_ar_order(), self.get_ma_order() + 1)
            S = np.matrix(np.zeros([r, 1]), copy=False)
            R = np.matrix(np.zeros([1, 1]), copy=False)
            self._state_space_representation = StateSpaceModel(F, G, Q, R, S)
        return self._state_space_representation

    def _compute_Q(self):
        r = max(self.get_ar_order(), self.get_ma_order() + 1)
        Q = np.matrix(np.zeros([r, r]), copy=False)
        Q[r - 1, r - 1] = self.get_sigma_sq()
        return Q

    def _compute_G(self):
        r = max(self.get_ar_order(), self.get_ma_order() + 1)
        G = np.matrix(np.zeros([1, r]), copy=False)
        for k in range(r):
            G[0, r - 1 - k] = self.get_theta(k)
        return G

    def _compute_F(self):
        r = max(self.get_ar_order(), self.get_ma_order() + 1)
        F = np.matrix(np.zeros([r, r]), copy=False)
        for k in range(1, r):
            F[k - 1, k] = 1
        for k in range(1, r + 1):
            F[r - 1, r - k] = self.get_phi(k)
        return F

    def summary(self):
        print('ARMA({}, {}) with Z_t ~ N(0, {})'.format(self.get_ar_order(),
              self.get_ma_order(),
              round(self.get_sigma_sq(), 2)))
        if self.get_ar_order() > 0:
            print('AR coefficients:')
        for k in range(1, self.get_ar_order() + 1):
            print('phi_{}: {}'.format(k, round(self.get_phi(k), 2)))
        if self.get_ma_order() > 0:
            print('MA coefficients:')
        for k in range(1, self.get_ma_order() + 1):
            print('theta_{}: {}'.format(k, round(self.get_theta(k), 2)))

    def get_ma_infty_coef(self, k):
        if k in self._ma_infty_coefs:
            return self._ma_infty_coefs[k]
        psi_k = self._calculate_ma_infty_coef(k)
        self._ma_infty_coefs[k] = psi_k
        return psi_k

    #Brockwell, Davis p. 91
    def _calculate_ma_infty_coef(self, j):
        if j < max(self._p, self._q + 1):
            return self.get_theta(j) + sum(self.get_phi(k) * self.get_ma_infty_coef(j - k)
                                           for k in range(1, j + 1))
        return sum(self.get_phi(k) * self.get_ma_infty_coef(j - k) for k in range(1, self._p + 1))

    def plot_impulse_response(self, maxlag=20):
        plt.plot([self.get_ma_infty_coef(k) for k in range(maxlag + 1)])

    #Brockwell, Davis p. 93
    def auto_cov_funct(self, lag):
        lag = abs(lag)
        if lag in self._acf:
            return self._acf[lag]
        if self.get_ar_order() == 0:
            acovf = self._auto_cov_funct_ma(lag)
            self._acf[lag] = acovf
            return acovf
        if lag <= self._p:
            rhs = self._sigma_sq * np.array([
                sum(self.get_theta(j) * self.get_ma_infty_coef(j - k)
                    for j in range(k, self._q + 1)) for k in range(self._p + 1)
            ])
            lhs = np.zeros((self._p + 1, self._p + 1))
            for t in range(self._p + 1):
                lhs[0][t] = self._get_ar_coeff(t)
            for k in range(1, self._p + 1):
                lhs[k][0] = self._get_ar_coeff(k)
                for t in range(1, self._p + 1):
                    lhs[k][t] = self._get_ar_coeff(t + k) + self._get_ar_coeff(k - t)
            for k, acf_k in enumerate(np.linalg.lstsq(lhs, rhs)[0]):
                self._acf[k] = acf_k
            return self._acf[lag]
        acovf = self._calculate_auto_cov_funct(lag)
        self._acf[lag] = acovf
        return acovf

    def _calculate_auto_cov_funct(self, lag):
        return sum(self.get_phi(k) * self.auto_cov_funct(lag - k) for k in range(1, self._p + 1))

    def _auto_cov_funct_ma(self, lag):
        if lag > self.get_ma_order():
            return 0
        return self._sigma_sq * sum(self.get_theta(j) * self.get_theta(j + lag)
                                    for j in range(self._q + 1))

    #Brockwell, Davis p. 172
    def get_innovation_coef(self, n, j):
        if j > n:
            raise ValueError('innovation coefficient not defined for j > n')
        if (n, j) in self._innovation_coefs:
            return self._innovation_coefs[(n, j)]
        coef = self._calculate_innovation_coef(n, j)
        self._innovation_coefs[(n, j)] = coef
        return coef

    #Brockwell, Davis p. 175, ch. 5.3
    def _calculate_innovation_coef(self, n, j):
        k = n - j
        return (
            self._kappa_w(n + 1, k + 1) -
            sum(self.get_innovation_coef(k, k - i) *
                self.get_innovation_coef(n, n - i) *
                self._get_innovations_error_w(i)
                for i in range(k))
        ) / self._get_innovations_error_w(k)

    def get_innovations_error(self, n):
        return self._get_innovations_error_w(n) * self.get_sigma_sq()

    #Brockwell, Davis p. 175, ch. 5.3
    def _get_innovations_error_w(self, n):
        if n in self._innovation_errors:
            return self._innovation_errors[n]
        error = self._calculate_innovation_error(n)
        self._innovation_errors[n] = error
        return error

    def get_r(self, n):
        return self._get_innovations_error_w(n)

    #get r_0, ... , r_n-1
    def get_first_rs(self, n):
        rs = np.ones(n)
        for j in range(n):
            r = self.get_r(j)
            #rs converge to 1, so no point to continue computation
            if round(r, 20) == 1:
                break
            rs[j] = r
        return rs

    def _calculate_innovation_error(self, n):
        return self._kappa_w(n + 1, n + 1) - sum(
            self.get_innovation_coef(n, n - j) ** 2 *
            self._get_innovations_error_w(j)
            for j in range(n)
        )

    #Brockwell, Davis p. 175
    def _kappa_w(self, i, j):
        if 1 <= min(i, j) and max(i, j) <= self._m:
            return self.auto_cov_funct(i - j) / self._sigma_sq
        if min(i, j) <= self._m and self._m < max(i, j) and max(i, j) <= 2 * self._m:
            return (
                self.auto_cov_funct(i - j) -
                sum(self.get_phi(r) * self.auto_cov_funct(r - abs(i - j))
                    for r in range(1, self._p + 1))
            ) / self._sigma_sq
        if min(i, j) > self._m:
            return sum(self.get_theta(r) * self.get_theta(r + abs(i - j))
                       for r in range(self._q + 1))
        return 0


#class for state space representation of a time series
class StateSpaceModel:
    def __init__(self, F, G, Q, R, S):
        #time invariant state space model (see Brockwell, Davis ch. 12):
        #Y_t = FX_t + W_t
        #X_t+1 = GX_t + V_t
        #covariance matrix of [W_t, V_t]' = [[Q, S'], [S, R]]
        self._F = F
        self._G = G
        self._Q = Q
        self._R = R
        self._S = S

        self._pred_psi = {1: np.zeros(Q.shape)}
        self._pred_pi = {1: Q}

    def get_F(self):
        return self._F

    def get_G(self):
        return self._G

    def get_Q(self):
        return self._Q

    def get_R(self):
        return self._R

    def get_S(self):
        return self._S

    #Brockwell, Davis p. 476, Sigma from prop. 12.2.2
    def get_error_cov_matrix(self, t):
        if t < 1:
            raise ValueError('Prediction error only defined for positive index')
        return self.get_pred_pi(t) - self.get_pred_psi(t)

    def get_pred_psi(self, t):
        if t not in self._pred_psi:
            self._pred_psi[t] = self._F * self.get_pred_psi(t - 1) * self._F.T +\
                self.get_pred_theta(t - 1) *\
                np.linalg.pinv(self.get_pred_delta(t - 1)) *\
                self.get_pred_theta(t - 1).T
        return self._pred_psi[t]

    def get_pred_pi(self, t):
        if t not in self._pred_pi:
            self._pred_pi[t] = self._F * self.get_pred_pi(t - 1) * self._F.T + self._Q
        return self._pred_pi[t]

    def get_pred_delta(self, t):
        return self._G * self.get_error_cov_matrix(t) * self._G.T + self._R

    def get_pred_theta(self, t):
        return self._F * self.get_error_cov_matrix(t) * self._G.T + self._S


#class for transformations and backtransformations
class Transform:
    def __init__(self, d=0, mean=0, box_cox=None):
        if d < 0:
            raise ValueError('can not have negative differencing coefficient!')
        if box_cox is not None:
            if box_cox < 0:
                raise ValueError('box cox transformation only defined for non negative lambda')
        self._d = d
        self._mean = mean
        self._box_cox = box_cox

    def transform(self, data):
        data = np.array(data)
        if self._box_cox is not None:
            data = self._box_cox_transform(data)
        data = self._subtract_mean(data, self._mean)
        return data

    def backtransform(self, data):
        data = np.array(data)
        data = self._add_mean(data, self._mean)
        if self._box_cox is not None:
            data = self._box_cox_backtransform(data)
        return data

    def is_mean_corrected(self):
        if self._mean == 0:
            return False
        else:
            return True

    def _add_mean(self, data, mean):
        return data + mean

    def _subtract_mean(self, data, mean):
        return data - mean

    #Brockwell, Davis p. 284
    def _box_cox_transform(self, data):
        if self._box_cox == 0:
            return np.log(data)
        return (data ** self._box_cox - 1) / self._box_cox

    def _box_cox_backtransform(self, data):
        if self._box_cox == 0:
            return np.exp(data)
        return (data * self._box_cox + 1) ** (1 / self._box_cox)
