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

        for k in range(20):
            self.assertAlmostEqual(model_332.auto_cov_funct(k), 3 * 2 ** -k * (32 / 3 + 8 * k))

        for k in range(4, 100):
            self.assertEqual(model_ma.auto_cov_funct(k), 0)

        self.assertEqual(model_empty.auto_cov_funct(0), 1)
        self.assertEqual(model_empty.auto_cov_funct(1), 0)
        self.assertEqual(model_empty.auto_cov_funct(10), 0)

    def test_innovation_coefs(self):
        model_arma11 = PureARMA([0.2], [0.4])
        model_empty = PureARMA(sigma_sq=4)

        self.assertAlmostEqual(model_arma11.get_innovation_coef(1, 1), 0.2909, 4)
        self.assertAlmostEqual(model_arma11.get_innovation_coef(2, 1), 0.3833, 4)
        for n in range(6, 11):
            self.assertAlmostEqual(model_arma11.get_innovation_coef(n, 1), 0.4, 4)

        self.assertEqual(model_empty.get_innovation_coef(1, 1), 0)
        self.assertEqual(model_empty.get_innovation_coef(9, 3), 0)

    def test_innovation_r(self):
        model_arma11 = PureARMA([0.2], [0.4])

        self.assertAlmostEqual(model_arma11.get_r(1), 1.0436, 4)
        self.assertAlmostEqual(model_arma11.get_r(2), 1.0067, 4)
        for n in range(6, 11):
            self.assertAlmostEqual(model_arma11.get_r(n), 1, 4)


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

if __name__ == '__main__':
    unittest.main()
