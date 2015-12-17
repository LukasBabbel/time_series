import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize

limit = 50


#class for handling the ugly side (sample (p)acf, model fitting...)
class ARMA:
    def __init__(self, data, d=0, subtract_mean=True, box_cox=None):
        self._mean = np.mean(data)
        mean = 0
        if subtract_mean:
            mean = self._mean

        self.set_transformation(Transform(d=d, mean=mean, box_cox=box_cox))

        self._data = self._transform(data)

        self._sample_auto_covariance = {}
        self.model = None

        self._implemented_ar_methods = ('durbin_levinson', 'closed_form', 'min_reduced_likelihood', 'least_squares')
        self._implemented_ma_methods = ('durbin_levinson', 'min_reduced_likelihood', 'least_squares')
        self._implemented_arma_methods = ('durbin_levinson', 'min_reduced_likelihood', 'least_squares')

    def set_transformation(self, transformation):
        self._transformation = transformation
        self._transform = transformation.transform
        self._backtransform = transformation.backtransform

    def sample_autocovariance(self, lag):
        lag = abs(lag)
        if lag in self._sample_auto_covariance:
            return self._sample_auto_covariance[lag]
        n = len(self._data)
        if lag >= n:
            raise ValueError('lag out of range')
        return (1.0 / n) * sum(self._data[-(n - lag):] * self._data[:n - lag])

    def sample_acf(self, lag):
        return self.sample_autocovariance(lag) / self.sample_autocovariance(0)

    #ToDo redo without append
    def sample_covariance_matrix(self, k):
        row = []
        for l in range(k):
            row.append(self.sample_autocovariance(l))
        matrix = []
        for l in range(k):
            matrix.append(row[1:1 + l][::-1] + row[:k - l])
        return np.matrix(matrix)

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
            raise ValueError('unknown method, implemented methods:' + str(self._implemented_ar_methods))

        if p == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            self.model = self._fit_ar_durbin_levinson(p)
        elif method == 'closed_form':
            self.model = self._fit_ar_closed_form(p)
        elif method == 'min_reduced_likelihood':
            self.model = self._fit_arma_max_likelihood(p=p)
        elif method == 'least_squares':
            self.model = self._fit_arma_least_squares(p=p)

    def _fit_ar_closed_form(self, p):
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
            raise ValueError('unknown method, implemented methods:' + str(self._implemented_ma_methods))

        if q == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            self.model = self._fit_ma_durbin_levinson(q)
        elif method == 'min_reduced_likelihood':
            self.model = self._fit_arma_max_likelihood(q=q)
        elif method == 'least_squares':
            self.model = self._fit_arma_least_squares(q=q)

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
            nu[m + 1] = self.sample_autocovariance(0) - sum(theta[m][m - j] ** 2 * nu[j] for j in range(m + 1))
        return PureARMA(theta=theta[q - 1], sigma_sq=nu[q])

    #ToDo write tests
    def get_confidence_interval_ma_d_l(self):
        return [1.96 *
                np.sqrt(sum(self.model.get_theta(k) ** 2 for k in range(j + 1))) /
                np.sqrt(len(self._data))
                for j in range(self.model.get_ma_order())]

    def fit_arma(self, p, q, method='durbin_levinson', **kwargs):
        if method not in self._implemented_arma_methods:
            raise ValueError('unknown method, implemented methods:' + str(self._implemented_arma_methods))

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

    #ToDo test this
    def _fit_arma_durbin_levinson(self, p, q, m=0):
        m = max(m, p + q)
        ma_model = self._fit_ma_durbin_levinson(m)
        estimation_matrix = np.matrix([[ma_model.get_theta(q + j - i) for i in range(p)] for j in range(p)])
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

    def get_training_predictions(self, model=None):
        return self.get_one_step_predictors(len(self._data), model)

    def get_one_step_predictor(self, n, model=None):
        return self.get_one_step_predictors(n, model)[n]

    def get_one_step_predictors(self, n, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        m = max(model.get_ar_order(), model.get_ma_order())
        if n > len(self._data):
            raise ValueError('One step prediction only possible for one step ahead')
        predictions = np.zeros(max(n, m) + 1)
        for k in range(1, m):
            predictions[k] = sum(
                model.get_innovation_coef(k, j) * (self._data[k - j] - predictions[k - j]) for j in range(1, k + 1)
            )
        for k in range(m, n + 1):
            k_limited = min(limit + m, k)
            predictions[k] = sum(
                model.get_phi(j) * self._data[k - j] for j in range(1, model.get_ar_order() + 1)
            ) + sum(
                model.get_innovation_coef(k_limited, j) * (self._data[k - j] - predictions[k - j]) for j in range(1, model.get_ma_order() + 1)
            )
        return predictions

    def get_weighted_sum_squared_residuals(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        predictions = self.get_training_predictions(model=model)
        rs = np.zeros(len(self._data))
        for i in range(len(self._data)):
            rs[i] = model.get_r(i)
        return np.sum((self._data - predictions[:-1]) ** 2 / rs)

    def _wsum_residuals_by_param(self, params, p, q):
        phi = params[:p]
        theta = params[p:p + q]
        sigma_sq = 1
        model = PureARMA(phi=phi, theta=theta, sigma_sq=sigma_sq)
        return self.get_weighted_sum_squared_residuals(model=model)

    def get_reduced_likelihood(self, model=None):
        if model is None:
            if self.model is None:
                raise ValueError('no model specified')
            else:
                model = self.model
        rs = np.zeros(len(self._data))
        for i in range(len(self._data)):
            rs[i] = model.get_r(i)
        return np.log(self.get_weighted_sum_squared_residuals(model=model) / len(self._data)) + np.sum(np.log(rs)) / len(self._data)

    def _reduced_likelihood_by_param(self, params, p, q):
        phi = params[:p]
        theta = params[p:p + q]
        sigma_sq = 1
        model = PureARMA(phi=phi, theta=theta, sigma_sq=sigma_sq)
        return self.get_reduced_likelihood(model=model)

    #ToDo testing and error handling if minimizations fails
    def _fit_arma_max_likelihood(self, p=0, q=0):
        start_params = self._calculate_initial_coeffs(p, q)

        def to_minimize(params):
                    return self._reduced_likelihood_by_param(params, p, q)
        opt_params = scipy.optimize.minimize(to_minimize, x0=start_params)['x']

        opt_phi = opt_params[:p]
        opt_theta = opt_params[p:p + q]
        opt_sigma_sq = self._wsum_residuals_by_param(opt_params, p, q) / len(self._data)
        return PureARMA(opt_phi, opt_theta, opt_sigma_sq)

    #ToDo testing and error handling if minimizations fails
    def _fit_arma_least_squares(self, p=0, q=0):
        start_params = self._calculate_initial_coeffs(p, q)

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
        return np.hstack(start_model.get_params())[:-1]


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

    def summary(self):
        print('ARMA({}, {}) with Z_t ~ N(0, {})'.format(self.get_ar_order(), self.get_ma_order(), round(self.get_sigma_sq(), 2)))
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
            return self.get_theta(j) + sum(self.get_phi(k) * self.get_ma_infty_coef(j - k) for k in range(1, j + 1))
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
                sum(self.get_theta(j) * self.get_ma_infty_coef(j - k) for j in range(k, self._q + 1)) for k in range(self._p + 1)
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
        return self._sigma_sq * sum(self.get_theta(j) * self.get_theta(j + lag) for j in range(self._q + 1))

    #Brockwell, Davis p. 172
    def get_innovation_coef(self, n, j):
        if j > n:
            raise ValueError('innovation coefficient not defined for j > n')
        if (n, j) in self._innovation_coefs:
            return self._innovation_coefs[(n, j)]
        coef = self._calculate_innovation_coef(n, j)
        self._innovation_coefs[(n, j)] = coef
        return coef

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

    def _get_innovations_error_w(self, n):
        if n in self._innovation_errors:
            return self._innovation_errors[n]
        error = self._calculate_innovation_error(n)
        self._innovation_errors[n] = error
        return error

    def get_r(self, n):
        return self._get_innovations_error_w(n)

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
                sum(self.get_phi(r) * self.auto_cov_funct(r - abs(i - j)) for r in range(1, self._p + 1))
            ) / self._sigma_sq
        if min(i, j) > self._m:
            return sum(self.get_theta(r) * self.get_theta(r + abs(i - j)) for r in range(self._q + 1))
        return 0


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

    def _add_mean(self, data, mean):
        return data + mean

    def _subtract_mean(self, data, mean):
        return data - mean

    def _box_cox_transform(self, data):
        if self._box_cox == 0:
            return np.log(data)
        return (data ** self._box_cox - 1) / self._box_cox

    def _box_cox_backtransform(self, data):
        if self._box_cox == 0:
            return np.exp(data)
        return (data * self._box_cox + 1) ** (1 / self._box_cox)
