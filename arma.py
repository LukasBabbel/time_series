import numpy as np
import matplotlib.pyplot as plt
import seaborn


#class for handling the ugly side (sample (p)acf, model fitting...)
class ARMA:
    def __init__(self, data):
        data = np.array(data)
        self._mean = data.mean()
        self._data = data - self._mean

        self._sample_auto_covariance = {}
        self.model = None

        self._implemented_ar_methods = ('durbin_levinson', 'closed_form')
        self._implemented_ma_methods = ('durbin_levinson')
        self._implemented_arma_methods = ('durbin_levinson')

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
        return [1.96 * np.sqrt((self.model.get_sigma_sq() * np.linalg.inv(self.sample_covariance_matrix(self.model.get_ar_order())))[j, j]) / np.sqrt(len(self._data)) for j in range(self.model.get_ar_order())]

    def fit_ma(self, q, method='durbin_levinson'):
        if method not in self._implemented_ma_methods:
            raise ValueError('unknown method, implemented methods:' + str(self._implemented_ma_methods))

        if q == 0:
            self.model = PureARMA(sigma_sq=self.sample_autocovariance(0))
        elif method == 'durbin_levinson':
            self.model = self._fit_ma_durbin_levinson(q)

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
        return [1.96 * np.sqrt(sum(self.model.get_theta(k) ** 2 for k in range(j + 1))) / np.sqrt(len(self._data)) for j in range(self.model.get_ma_order())]

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

    #test this
    def _fit_arma_durbin_levinson(self, p, q, m=0):
        m = max(m, p + q)
        ma_model = self._fit_ma_durbin_levinson(m)
        estimation_matrix = np.matrix([[ma_model.get_theta(q + j - i) for i in range(p)] for j in range(p)])
        estimation_matrix = np.linalg.inv(estimation_matrix)
        phi = estimation_matrix * np.matrix([ma_model.get_theta(q + i + 1) for i in range(p)]).T
        theta = []
        for j in range(1, q + 1):
            theta_j = ma_model.get_theta(j)
            for i in range(1, min(j, p) + 1):
                if i == j:
                    theta_j -= phi[j - 1]
                else:
                    theta_j -= phi[i - 1] * a_model.get_theta(j - i)
            theta.append(theta_j)
        return PureARMA(phi, theta, ma_model.get_sigma_sq())

    def get_reduced_likelyhood(self, phi, theta, sigma_sq):
        thetas, nus = self.innovations_algorithm(phi, theta)
        x_hats = self.get_innovations(thetas, phi, theta)
        rs = nus / sigma_sq
        return np.log(self.weighted_sum_squares(x_hats, map(lambda x: 1 / x, rs)) / len(self._data)) + sum(rs) / len(self._data)

    def innovations_algorithm(self):
        nus = np.zeros(len(self._data))
        thetas = [np.zeros(k + 1) for k in range(len(self._data))]
        for n in range(len(self._data)):
            for k in range(n):
                thetas[n - 1][n - k - 1] = (self.sample_autocovariance(n - k) -
                                            sum(thetas[k - 1][k - j - 1] * thetas[n - 1][n - j - 1] * nus[j] for j in range(k))
                                            ) / nus[k]
            nus[n] = self.sample_autocovariance(0) - sum(thetas[n - 1][n - j - 1] ** 2 * nus[j] for j in range(n))
        return thetas, nus

    #ToDo
    def get_innovations(self, thetas, phi, theta):
        return None

    def weighted_sum_squares(self, values, weights):
        return np.sum(values ** 2 * weights)


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
        if lag <= self._p:
            rhs = self._sigma_sq * np.array([sum(self.get_theta(j) * self.get_ma_infty_coef(j - k) for j in range(k, self._q + 1)) for k in range(self._p + 1)])
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
        return sum(self.get_phi(k) * self.auto_cov_funct(lag - k) for k in range(1, self._p + 1))

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
            sum(self.get_innovation_coef(k, k - i) * self.get_innovation_coef(n, n - i) * self.get_innovations_error(i) for i in range(k))
        ) / self.get_innovations_error(k)

    def get_innovations_error(self, n):
        if n in self._innovation_errors:
            return self._innovation_errors[n]
        error = self._calculate_innovation_error(n)
        self._innovation_errors[n] = error
        return error

    def get_r(self, n):
        return self.get_innovations_error(n) / self._sigma_sq

    def _calculate_innovation_error(self, n):
        return self._kappa_w(n + 1, n + 1) - sum(self.get_innovation_coef(n, n - j) ** 2 * self.get_innovations_error(j) for j in range(n))

    #Brockwell, Davis p. 175
    def _kappa_w(self, i, j):
        if 1 <= min(i, j) and max(i, j) <= self._m:
            return self.auto_cov_funct(i - j) / self._sigma_sq
        if min(i, j) <= self._m and self._m < max(i, j) and max(i, j) <= 2 * self._m:
            return (
                self.auto_cov_funct(i - j) -
                sum(phi_r * self.auto_cov_funct(r - abs(i - j) + 1) for r, phi_r in enumerate(self._phi))
            ) / self._sigma_sq
        if min(i, j) > self._m:
            return sum(self.get_theta(r) * self.get_theta(r + abs(i - j)) for r in range(self._q + 1))
        return 0
