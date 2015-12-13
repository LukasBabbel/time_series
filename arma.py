import numpy as np
import matplotlib.pyplot as plt
import seaborn


class ARMA:
    def __init__(self, data):
        data = np.array(data)
        self._mean = data.mean()
        self._data = data - self._mean

        self._sample_auto_covariance = {}
        self.model = None

    def sample_autocovariance(self, lag):
        lag = abs(lag)
        if lag in self._sample_auto_covariance:
            return self._sample_auto_covariance[lag]
        n = len(self._data)
        if lag >= n:
            raise ValueError('lag out of range')
        return (1.0 / n) * sum(self._data[-(n - lag):] * self._data[:n - lag])

    def sample_autococorrelation_function(self, lag):
        return self.sample_autocovariance(lag) / self.sample_autocovariance(0)

    def sample_covariance_matrix(self, k):
        row = []
        for l in range(k):
            row.append(self.sample_autocovariance(l))
        matrix = []
        for l in range(k):
            matrix.append(row[1:1 + l][::-1] + row[:k - l])
        return np.matrix(matrix)

    def partial_autocorrelation_function(self, k):
        if k == 0:
            return 1
        Gamma_k = self.sample_covariance_matrix(k)
        gamma_k = np.matrix([self.sample_autocovariance(l) for l in range(1, k + 1)]).T
        return (np.linalg.inv(Gamma_k) * gamma_k).item((k - 1, 0))

    def plot_ACF(self, limit=None):
        if not limit:
            limit = len(self._data)
        AC = []
        for lag in range(1, limit):
            AC.append(self.sample_autococorrelation_function(lag))
        plt.bar(list(range(1, limit)), AC)
        plt.axhline(1.96 / np.sqrt(len(self.data)), linestyle='--', alpha=0.6)
        plt.axhline(-1.96 / np.sqrt(len(self.data)), linestyle='--', alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('ACF')

    def plot_PACF(self, limit=None):
        if not limit:
            limit = len(self._data)
        PAC = []
        for lag in range(1, limit):
            PAC.append(self.partial_autocorrelation_function(lag))
        plt.bar(list(range(1, limit)), PAC)
        plt.axhline(1.96 / np.sqrt(len(self.data)), linestyle='--', alpha=0.6)
        plt.axhline(-1.96 / np.sqrt(len(self.data)), linestyle='--', alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('PACF')

    def fit_ar(self, p):
        Gamma = self.sample_covariance_matrix(p)
        gamma = np.matrix([self.sample_autocovariance(l) for l in range(1, p + 1)]).T
        self.coefs = np.linalg.inv(Gamma) * gamma
        self.sigma = self.sample_autocovariance(0) - self.coefs.T * gamma

    def fit_ar_durbin_levinson(self, p):
        self.nu = np.zeros(p)
        self.phi = [np.zeros(m + 1) for m in range(p)]
        self.nu[0] = self.sample_autocovariance(0) * (1 - self.sample_autococorrelation_function(1) ** 2)
        self.phi[0][0] = self.sample_autococorrelation_function(1)
        for m in range(1, p):
            self.phi[m][m] = (self.sample_autocovariance(m + 1) - sum(self.phi[m - 1][j] * self.sample_autocovariance(m - j) for j in range(m))) / self.nu[m - 1]
            for j in range(m):
                self.phi[m][j] = self.phi[m - 1][j] - self.phi[m][m] * self.phi[m - 1][m - 1 - j]
            self.nu[m] = self.nu[m - 1] * (1 - self.phi[m][m] ** 2)

    def get_confidence_interval_ar_d_l(self, p):
        return [1.96 * np.sqrt((self.nu[p - 1] * np.linalg.inv(self.sample_covariance_matrix(p)))[j, j]) / np.sqrt(len(self.data)) for j in range(p)]

    def fit_ma_durbin_levinson(self, q):
        self.nu_ma = np.zeros(q + 1)
        self.theta_ma = [np.zeros(m + 1) for m in range(q)]
        self.nu_ma[0] = self.sample_autocovariance(0)
        for m in range(q):
            for k in range(m + 1):
                self.theta_ma[m][m - k] = (self.sample_autocovariance(m - k + 1) - sum(self.theta_ma[m][m - j] * self.theta_ma[k - 1][k - j - 1] * self.nu_ma[j] for j in range(k))) / self.nu_ma[k]
            self.nu_ma[m + 1] = self.sample_autocovariance(0) - sum(self.theta_ma[m][m - j] ** 2 * self.nu_ma[j] for j in range(m + 1))

    def get_confidence_interval_ma_d_l(self, q):
        return [1.96 * np.sqrt(sum(self.theta_ma[q - 1][k] ** 2 for k in range(j + 1))) / np.sqrt(len(self.data)) for j in range(q)]

    def fit_arma_preliminary(self, p, q, m=0):
        m = max(m, p + q)
        self.fit_ma_durbin_levinson(m)
        estimation_matrix = np.matrix([[self.theta_ma[m - 1][q + j - i - 1] for i in range(p)] for j in range(p)])
        estimation_matrix = np.linalg.inv(estimation_matrix)
        self.phi = estimation_matrix * np.matrix([self.theta_ma[m - 1][q + i] for i in range(p)]).T
        self.theta = []
        for j in range(1, q + 1):
            theta_j = self.theta_ma[m - 1][j - 1]
            for i in range(1, min(j, p) + 1):
                if i == j:
                    theta_j -= self.phi[j - 1]
                else:
                    theta_j -= self.phi[i - 1] * self.theta_ma[m - 1][j - i - 1]
            self.theta.append(theta_j)
        self.theta = np.array(self.theta)

    def get_reduced_likelyhood(self, phi, theta, sigma_sq):
        thetas, nus = self.innovations_algorithm(phi, theta)
        x_hats = self.get_innovations(thetas, phi, theta)
        rs = nus / sigma_sq
        return np.log(self.weighted_sum_squares(x_hats, map(lambda x: 1 / x, rs)) / len(self.data)) + sum(rs) / len(self.data)

    def innovations_algorithm(self):
        nus = np.zeros(len(self.data))
        thetas = [np.zeros(k + 1) for k in range(len(self.data))]
        for n in range(len(self.data)):
            for k in range(n):
                thetas[n - 1][n - k - 1] = (self.sample_autocovariance(n - k) -
                                            sum(thetas[k - 1][k - j - 1] * thetas[n - 1][n - j - 1] * nus[j] for j in range(k))) / nus[k]
            nus[n] = self.sample_autocovariance(0) - sum(thetas[n - 1][n - j - 1] ** 2 * nus[j] for j in range(n))
        return thetas, nus

    #ToDo
    def get_innovations(self, thetas, phi, theta):
        return None

    def weighted_sum_squares(self, values, weights):
        return sum(value ** 2 * weight for value, weight in zip(values, weights))


#class for handling the pure math side, not supposed to see data
class PureARMA:
    def __init__(self, phi=None, theta=None, sigma_sq=1):
        if phi is None:
            phi = []
        if theta is None:
            theta = []
        self._phi = phi
        self._theta = theta
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

    def get_phi(self, k):
        if k == 0:
            return 1
        if k <= self._p:
            return self._phi[k - 1]
        return 0

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
