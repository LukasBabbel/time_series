import unittest
import numpy as np
from arma import PureARMA, ARMA


def simulate_arma(phi=None, theta=None, sigma=1, simulations=1000):
    if phi is None:
        phi = []
    if theta is None:
        theta = []
    phi = phi
    theta = [1] + theta
    Z = [np.random.normal(scale=sigma) for k in range(simulations + len(phi) + len(theta))]
    X = []
    for k in range(len(phi)):
        X.append(sum(a * b for a, b in zip(theta, Z[k:k + len(theta)])))
    for k in range(len(phi), simulations + len(phi)):
        X.append(sum(a * b for a, b in zip(theta, Z[k:k + len(theta)])) + sum(a * b for a, b in zip(phi, X[::-1][:len(phi)])))
    return np.array(X[len(phi):])


class Test_PureARMA(unittest.TestCase):
    def test_parameters(self):
        sigma_sq = 2
        model = PureARMA([-1, 2, 3], [4, 5], sigma_sq)

        self.assertEqual(model.get_phi(2), 2)
        self.assertEqual(model.get_phi(0), 1)
        self.assertEqual(model.get_phi(4), 0)

        self.assertEqual(model.get_theta(2), 5)
        self.assertEqual(model.get_theta(0), 1)
        self.assertEqual(model.get_theta(3), 0)

        self.assertEqual(model.get_sigma_sq(), sigma_sq)

    def test_ma_infty(self):
        model_ma = PureARMA([], [1, 2, 3])
        model_ar = PureARMA([0.5])
        model_empty = PureARMA()

        self.assertEqual(model_ma.get_ma_infty_coef(0), 1)
        self.assertEqual(model_ma.get_ma_infty_coef(2), 2)
        self.assertEqual(model_ma.get_ma_infty_coef(3), 3)
        self.assertEqual(model_ma.get_ma_infty_coef(5), 0)
        self.assertEqual(model_ma.get_ma_infty_coef(3), 3)

        self.assertEqual(model_ar.get_ma_infty_coef(0), 1)
        self.assertEqual(model_ar.get_ma_infty_coef(1), 0.5)
        self.assertEqual(model_ar.get_ma_infty_coef(2), 0.25)
        self.assertEqual(model_ar.get_ma_infty_coef(50), 0.5 ** 50)

        self.assertEqual(model_empty.get_ma_infty_coef(0), 1)
        self.assertEqual(model_empty.get_ma_infty_coef(1), 0)
        self.assertEqual(model_empty.get_ma_infty_coef(10), 0)

    def test_acf(self):
        model_332 = PureARMA([1, -0.25], [1], 3)
        model_ma = PureARMA([], [1, 2, 3])
        model_empty = PureARMA()
        model_arma23 = PureARMA(phi=[1, -0.24], theta=[0.4, 0.2, 0.1])
        arma_model = PureARMA(phi=[0.7], theta=[0.3], sigma_sq=4)

        for k in range(20):
            self.assertAlmostEqual(model_332.auto_cov_funct(k), 3 * 2 ** -k * (32 / 3 + 8 * k))

        for k in range(4, 100):
            self.assertEqual(model_ma.auto_cov_funct(k), 0)

        self.assertEqual(model_empty.auto_cov_funct(0), 1)
        self.assertEqual(model_empty.auto_cov_funct(1), 0)
        self.assertEqual(model_empty.auto_cov_funct(10), 0)

        self.assertAlmostEqual(model_arma23.auto_cov_funct(0), 7.17133, 5)
        self.assertAlmostEqual(model_arma23.auto_cov_funct(1), 6.44139, 5)
        self.assertAlmostEqual(model_arma23.auto_cov_funct(2), 5.06027, 5)

        self.assertAlmostEqual(arma_model.auto_cov_funct(0), 4 * (1 + 2 * 0.3 * 0.7 + 0.3 ** 2) / (1 - 0.7 ** 2), 5)

    def test_innovation_coefs(self):
        model_arma11 = PureARMA([0.2], [0.4])
        model_empty = PureARMA(sigma_sq=4)
        model_arma23 = PureARMA(phi=[1, -0.24], theta=[0.4, 0.2, 0.1])

        self.assertAlmostEqual(model_arma11.get_innovation_coef(1, 1), 0.2909, 4)
        self.assertAlmostEqual(model_arma11.get_innovation_coef(2, 1), 0.3833, 4)
        for n in range(6, 11):
            self.assertAlmostEqual(model_arma11.get_innovation_coef(n, 1), 0.4, 4)

        self.assertEqual(model_empty.get_innovation_coef(1, 1), 0)
        self.assertEqual(model_empty.get_innovation_coef(9, 3), 0)

        self.assertAlmostEqual(model_arma23.get_innovation_coef(1, 1), 0.8982, 4)
        self.assertAlmostEqual(model_arma23.get_innovation_coef(2, 1), 1.3685, 4)
        self.assertAlmostEqual(model_arma23.get_innovation_coef(2, 2), 0.7056, 4)

    def test_innovation_r(self):
        model_arma11 = PureARMA([0.2], [0.4])
        model_ma = PureARMA(theta=[0.5], sigma_sq=3)
        model_arma23 = PureARMA(phi=[1, -0.24], theta=[0.4, 0.2, 0.1])

        self.assertAlmostEqual(model_arma11.get_r(0), 1.375, 4)
        self.assertAlmostEqual(model_arma11.get_r(1), 1.0436, 4)
        self.assertAlmostEqual(model_arma11.get_r(2), 1.0067, 4)
        for n in range(6, 11):
            self.assertAlmostEqual(model_arma11.get_r(n), 1, 4)

        self.assertAlmostEqual(model_ma.get_r(0), (1 + 0.5 ** 2) / 3, 4)

        self.assertAlmostEqual(model_arma23.get_r(0), 7.1713, 4)
        self.assertAlmostEqual(model_arma23.get_r(1), 1.3856, 4)
        self.assertAlmostEqual(model_arma23.get_r(2), 1.0057, 4)

    def test_kappa_w(self):
        theta = 0.4
        sigma_sq = 3
        ma_model = PureARMA(theta=[theta], sigma_sq=sigma_sq)
        arma_model = PureARMA(phi=[0.7], theta=[0.3], sigma_sq=4)
        model_arma23 = PureARMA(phi=[1, -0.24], theta=[0.4, 0.2, 0.1])

        self.assertAlmostEqual(ma_model._kappa_w(1, 1), 1 + theta ** 2)
        self.assertAlmostEqual(ma_model._kappa_w(2, 2), 1 + theta ** 2)
        self.assertAlmostEqual(ma_model._kappa_w(3, 3), 1 + theta ** 2)
        self.assertAlmostEqual(ma_model._kappa_w(1, 3), 0)
        self.assertAlmostEqual(ma_model._kappa_w(1, 2), theta)

        self.assertAlmostEqual(arma_model._kappa_w(1, 1), (1 + 2 * 0.3 * 0.7 + 0.3 ** 2) / (1 - 0.7 ** 2))
        self.assertAlmostEqual(arma_model._kappa_w(2, 2), 1 + 0.3 ** 2)
        self.assertAlmostEqual(arma_model._kappa_w(3, 3), 1 + 0.3 ** 2)
        self.assertAlmostEqual(arma_model._kappa_w(1, 3), 0)
        self.assertAlmostEqual(arma_model._kappa_w(1, 4), 0)
        self.assertAlmostEqual(arma_model._kappa_w(1, 5), 0)
        self.assertAlmostEqual(arma_model._kappa_w(20, 30), 0)
        self.assertAlmostEqual(arma_model._kappa_w(1, 2), 0.3)
        self.assertAlmostEqual(arma_model._kappa_w(3, 2), 0.3)
        self.assertAlmostEqual(arma_model._kappa_w(16, 15), 0.3)

        self.assertAlmostEqual(model_arma23._kappa_w(1, 1), 7.17133, 5)
        self.assertAlmostEqual(model_arma23._kappa_w(1, 2), 6.44139, 5)
        self.assertAlmostEqual(model_arma23._kappa_w(1, 3), 5.06027, 5)
        self.assertAlmostEqual(model_arma23._kappa_w(4, 7), 0.1, 5)
        self.assertAlmostEqual(model_arma23._kappa_w(1, 5), 0, 5)
        self.assertAlmostEqual(model_arma23._kappa_w(2, 2), 7.17133, 5)


class Test_ARMA(unittest.TestCase):
    def test_sample_autocovariance(self):
        zeros = np.zeros(100)
        zero_ts = ARMA(zeros)

        for k in range(100):
            self.assertEqual(zero_ts.sample_autocovariance(k), 0)

    def test_ar_fit_closed_form(self):
        zeros = np.zeros(100)
        zero_ts = ARMA(zeros)

        np.random.seed(12345)
        simulated_ar3 = simulate_arma(phi=[0.5, 0.1, 0.2], sigma=0.5)
        ar3 = ARMA(simulated_ar3)

        zero_ts.fit_ar(p=0, method='closed_form')
        self.assertEqual(len(zero_ts.model.get_params()[0]), 0)

        ar3.fit_ar(p=3, method='closed_form')
        self.assertAlmostEqual(ar3.model.get_params()[0][0], 0.5, 1)
        self.assertAlmostEqual(ar3.model.get_params()[0][1], 0.1, 1)
        self.assertAlmostEqual(ar3.model.get_params()[0][2], 0.2, 1)

    def test_ar_fit_durbin_levinson(self):
        np.random.seed(12345)
        simulated_ar3 = simulate_arma(phi=[0.5, 0.1, 0.2], sigma=0.5)
        ar3 = ARMA(simulated_ar3)

        ar3.fit_ar(p=3, method='durbin_levinson')
        self.assertAlmostEqual(ar3.model.get_phi(1), 0.5, 1)
        self.assertAlmostEqual(ar3.model.get_phi(2), 0.1, 1)
        self.assertAlmostEqual(ar3.model.get_phi(3), 0.2, 1)
        self.assertAlmostEqual(ar3.model.get_sigma_sq(), 0.25, 1)

    def test_ma_fit_durbin_levinson(self):
        np.random.seed(12345)
        simulated_ma1 = simulate_arma(theta=[0], sigma=0.5, simulations=10000)
        ma1 = ARMA(simulated_ma1)

        ma1.fit_ma(q=1, method='durbin_levinson')
        self.assertAlmostEqual(ma1.model.get_theta(1), 0, 1)
        self.assertAlmostEqual(ma1.model.get_sigma_sq(), 0.25, 1)

    def test_one_step_predictions(self):
        ma_model = PureARMA(theta=[-0.9], sigma_sq=1)
        ma_data = [-2.58, 1.62, -0.96, 2.62, -1.36]
        ma = ARMA(ma_data)
        ma._data = ma._data + ma._mean

        self.assertEqual(ma.get_one_step_predictor(0, ma_model), 0)
        self.assertAlmostEqual(ma.get_one_step_predictor(1, ma_model), 1.28, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(2, ma_model), -0.22, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(3, ma_model), 0.55, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(4, ma_model), -1.63, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(5, ma_model), -0.22, 2)

        arma_data = [-1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228, -1.093]
        arma_model = PureARMA(phi=[0.2], theta=[0.4], sigma_sq=1)
        arma = ARMA(arma_data)
        arma._data = arma._data + arma._mean

        self.assertEqual(arma.get_one_step_predictor(0, arma_model), 0)
        self.assertAlmostEqual(arma.get_one_step_predictor(1, arma_model), -0.534, 1)
        self.assertAlmostEqual(arma.get_one_step_predictor(2, arma_model), 0.5068, 1)
        self.assertAlmostEqual(arma.get_one_step_predictor(3, arma_model), -0.1321, 1)
        self.assertAlmostEqual(arma.get_one_step_predictor(4, arma_model), -0.4539, 1)
        self.assertAlmostEqual(arma.get_one_step_predictor(5, arma_model), 0.7046, 1)

        data = [1, 2, 3, 4, 5]
        empty_model = PureARMA()
        empty = ARMA(data)
        empty.model = empty_model

        for k in range(6):
            self.assertEqual(empty.get_one_step_predictor(k), 0)

    def test_sum_squared(self):
        values = np.array([1, 2, 3, 4])
        weights = np.array([2, 1, 1, 3])
        empty = np.array([])
        arma = ARMA([1, 2])

        self.assertEqual(arma.weighted_sum_squares(values, weights), 2 + 4 + 9 + 16 * 3)
        self.assertEqual(arma.weighted_sum_squares(empty, empty), 0)


if __name__ == '__main__':
    unittest.main()
