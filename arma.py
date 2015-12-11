import numpy as np
import matplotlib.pyplot as plt
import seaborn


class ARMA:
    def __init__(self, data):
        data = np.array(data)
        self.mean = data.mean()
        self.data = data - self.mean

        #cache first 100 values of acf
        self._acf = [None] * 100
        self._acf = [self.sample_autocovariance_function(k) for k in range(min(100, len(self.data)))]

    def sample_autocovariance_function(self, lag):
        lag = abs(lag)
        if lag < min(100, len(self.data)) and self.acf[lag] is not None:
            return self._acf[lag]
        n = len(self.data)
        if lag >= n:
            raise ValueError('lag out of range')
        return (1.0 / n) * sum(self.data[-(n - lag):] * self.data[:n - lag])

    def sample_autococorrelation_function(self, lag):
        return self.sample_autocovariance_function(lag) / self.sample_autocovariance_function(0)

    def sample_covariance_matrix(self, k):
        row = []
        for l in range(k):
            row.append(self.sample_autocovariance_function(l))
        matrix = []
        for l in range(k):
            matrix.append(row[1:1 + l][::-1] + row[:k - l])
        return np.matrix(matrix)

    def partial_autocorrelation_function(self, k):
        if k == 0:
            return 1
        Gamma_k = self.sample_covariance_matrix(k)
        gamma_k = np.matrix([self.sample_autocovariance_function(l) for l in range(1, k + 1)]).T
        return (np.linalg.inv(Gamma_k) * gamma_k).item((k - 1, 0))

    def plot_ACF(self, limit=None):
        if not limit:
            limit = len(self.data)
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
            limit = len(self.data)
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
        gamma = np.matrix([self.sample_autocovariance_function(l) for l in range(1, p + 1)]).T
        self.coefs = np.linalg.inv(Gamma) * gamma
        self.sigma = self.sample_autocovariance_function(0) - self.coefs.T * gamma

    def fit_ar_durbin_levinson(self, p):
        self.nu = np.zeros(p)
        self.phi = [np.zeros(m + 1) for m in range(p)]
        self.nu[0] = self.sample_autocovariance_function(0) * (1 - self.sample_autococorrelation_function(1) ** 2)
        self.phi[0][0] = self.sample_autococorrelation_function(1)
        for m in range(1, p):
            self.phi[m][m] = (self.sample_autocovariance_function(m + 1) - sum(self.phi[m - 1][j] * self.sample_autocovariance_function(m - j) for j in range(m))) / self.nu[m - 1]
            for j in range(m):
                self.phi[m][j] = self.phi[m - 1][j] - self.phi[m][m] * self.phi[m - 1][m - 1 - j]
            self.nu[m] = self.nu[m - 1] * (1 - self.phi[m][m] ** 2)

    def get_confidence_interval_ar_d_l(self, p):
        return [1.96 * np.sqrt((self.nu[p - 1] * np.linalg.inv(self.sample_covariance_matrix(p)))[j, j]) / np.sqrt(len(self.data)) for j in range(p)]

    def fit_ma_durbin_levinson(self, q):
        self.nu_ma = np.zeros(q + 1)
        self.theta_ma = [np.zeros(m + 1) for m in range(q)]
        self.nu_ma[0] = self.sample_autocovariance_function(0)
        for m in range(q):
            for k in range(m + 1):
                self.theta_ma[m][m - k] = (self.sample_autocovariance_function(m - k + 1) - sum(self.theta_ma[m][m - j] * self.theta_ma[k - 1][k - j - 1] * self.nu_ma[j] for j in range(k))) / self.nu_ma[k]
            self.nu_ma[m + 1] = self.sample_autocovariance_function(0) - sum(self.theta_ma[m][m - j] ** 2 * self.nu_ma[j] for j in range(m + 1))

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
                thetas[n - 1][n - k - 1] = (self.sample_autocovariance_function(n - k) -
                                            sum(thetas[k - 1][k - j - 1] * thetas[n - 1][n - j - 1] * nus[j] for j in range(k))) / nus[k]
            nus[n] = self.sample_autocovariance_function(0) - sum(thetas[n - 1][n - j - 1] ** 2 * nus[j] for j in range(n))
        return thetas, nus

    #ToDo
    def get_innovations(self, thetas, phi, theta):
        return None

    def weighted_sum_squares(self, values, weights):
        return sum(value * weight for value, weight in zip(values, weights))


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
        self._ma_infty_coefs = {}
        self._acf = {}

    def get_params(self):
        return (self._phi, self._theta, self._sigma_sq)

    def get_phi(self, k):
        if k == 0:
            return 1
        if k <= self._p:
            return self._phi[k - 1]
        return 0

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

    #ToDo
    def acf(self, lag):
        lag = abs(lag)
        if lag in self._acf:
            return self._acf[lag]
        if lag < max(self._p, self._q + 1):
            rhs = self._sigma_sq * np.matrix([self.get_theta(j) * self.get_ma_infty_coef(j - lag) for j in range(lag, self._q + 1)])
            row = [1] + [-self.get_phi(__) for __ in range(1, self._p + 1)]
            lhs = np.matrix([row[-__:] + row[:-__] for __ in range(len(row))])
            for k, acf_k in np.linalg.inv(lhs) * rhs.T:
                self._acf[k] = acf_k
            return self._acf[lag]
        return sum(self.get_phi(k) * self.acf(lag - k) for k in range(1, self._p + 1))

    #Brockwell, Davis p. 175
    def _kappa_w(self, i, j):
        if 1 <= min(i, j) and max(i, j) <= self._m:
            return self.acf(i - j) / self._sigma_sq
        if min(i, j) <= self._m and self._m < max(i, j) and max(i, j) <= 2 * self._m:
            return (self.acf(i - j) - sum(phi_r * self.acf(r - abs(i - j) + 1) for r, phi_r in enumerate(self._phi))) / self._sigma_sq
        if min(i, j) > m:
            return sum(self.get_theta(r) * self.get_theta(r + abs(i - j)) for r in range(self._q + 1))
        return 0
