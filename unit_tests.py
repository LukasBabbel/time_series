import unittest
import numpy as np
from arma import PureARMA, ARMA, Transform


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

    def test_innovation_error(self):
        model_ma = PureARMA(theta=[-0.9], sigma_sq=1)

        self.assertAlmostEqual(model_ma.get_innovations_error(0), 1.81, 3)
        self.assertAlmostEqual(model_ma.get_innovations_error(1), 1.362, 3)
        self.assertAlmostEqual(model_ma.get_innovations_error(2), 1.215, 3)
        self.assertAlmostEqual(model_ma.get_innovations_error(3), 1.144, 3)
        self.assertAlmostEqual(model_ma.get_innovations_error(4), 1.102, 3)
        self.assertAlmostEqual(model_ma.get_innovations_error(5), 1.075, 3)

    def test_innovation_r(self):
        model_arma11 = PureARMA([0.2], [0.4])
        model_ma = PureARMA(theta=[0.5], sigma_sq=3)
        model_arma23 = PureARMA(phi=[1, -0.24], theta=[0.4, 0.2, 0.1])

        self.assertAlmostEqual(model_arma11.get_r(0), 1.375, 4)
        self.assertAlmostEqual(model_arma11.get_r(1), 1.0436, 4)
        self.assertAlmostEqual(model_arma11.get_r(2), 1.0067, 4)
        self.assertAlmostEqual(model_arma11.get_r(3), 1.0011, 4)
        self.assertAlmostEqual(model_arma11.get_r(4), 1.0002, 4)
        self.assertAlmostEqual(model_arma11.get_r(5), 1.0000, 4)
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
        ma = ARMA(ma_data, subtract_mean=False)

        self.assertEqual(ma.get_one_step_predictor(0, ma_model), 0)
        self.assertAlmostEqual(ma.get_one_step_predictor(1, ma_model), 1.28, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(2, ma_model), -0.22, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(3, ma_model), 0.55, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(4, ma_model), -1.63, 2)
        self.assertAlmostEqual(ma.get_one_step_predictor(5, ma_model), -0.22, 2)

        arma_data = [-1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228, -1.093]
        arma_model = PureARMA(phi=[0.2], theta=[0.4], sigma_sq=1)
        arma = ARMA(arma_data, subtract_mean=False)

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

    def test_weighted_sum_squared_residuals(self):
        data = [-2, 0, 2]
        arma = ARMA(data)
        model = PureARMA()
        ma_model = PureARMA(theta=[-0.9], sigma_sq=1)
        ma_data = [-2.58, 1.62, -0.96, 2.62, -1.36]
        ma = ARMA(ma_data, subtract_mean=False)

        self.assertAlmostEqual(arma.get_weighted_sum_squared_residuals(model), 8)

        self.assertAlmostEqual(ma.get_weighted_sum_squared_residuals(ma_model), 8.02, 1)

    def test_reduced_likelihood(self):
        ma_model = PureARMA(theta=[-0.9], sigma_sq=1)
        ma_data = [-2.58, 1.62, -0.96, 2.62, -1.36]
        ma = ARMA(ma_data)
        ma._data = ma._data + ma._mean

        self.assertAlmostEqual(ma.get_reduced_likelihood(ma_model), 0.7387, 1)

    def test_transform_mean(self):
        data = [-1, 0.5, 0, -0.5, 1]
        empty = []
        mean_zero = ARMA([1, -1])
        pos_mean = ARMA([2])
        neg_fraction_mean = ARMA([-0.5])

        self.assertListEqual(mean_zero._transform(data).tolist(), data)
        self.assertListEqual(pos_mean._transform(data).tolist(), [-3, -1.5, -2, -2.5, -1])
        self.assertListEqual(neg_fraction_mean._transform(data).tolist(), [-0.5, 1, 0.5, 0, 1.5])

        self.assertListEqual(mean_zero._transform(empty).tolist(), empty)
        self.assertListEqual(pos_mean._transform(empty).tolist(), empty)
        self.assertListEqual(neg_fraction_mean._transform(empty).tolist(), empty)

    def test_back_transform_mean(self):
        data = [-1, 0.5, 0, -0.5, 1]
        empty = []
        mean_zero = ARMA([1, -1])
        pos_mean = ARMA([2])
        neg_fraction_mean = ARMA([-0.5])

        self.assertListEqual(mean_zero._backtransform(data).tolist(), data)
        self.assertListEqual(pos_mean._backtransform(data).tolist(), [1, 2.5, 2, 1.5, 3])
        self.assertListEqual(neg_fraction_mean._backtransform(data).tolist(), [-1.5, 0, -0.5, -1, 0.5])

    def test_transform_backtransform(self):
        data = [-1, 0.5, 0, -0.5, 1]
        empty = []
        mean_zero = ARMA([1, -1])
        pos_mean = ARMA([2])
        neg_fraction_mean = ARMA([-0.5])

        self.assertListEqual(mean_zero._backtransform(mean_zero._transform(data)).tolist(), data)
        self.assertListEqual(pos_mean._backtransform(pos_mean._transform(data)).tolist(), data)
        self.assertListEqual(neg_fraction_mean._backtransform(neg_fraction_mean._transform(data)).tolist(), data)

        self.assertListEqual(mean_zero._backtransform(mean_zero._transform(empty)).tolist(), empty)
        self.assertListEqual(pos_mean._backtransform(pos_mean._transform(empty)).tolist(), empty)
        self.assertListEqual(neg_fraction_mean._backtransform(neg_fraction_mean._transform(empty)).tolist(), empty)


class Test_Transform(unittest.TestCase):
    def test_init(self):
        transf = Transform(d=3, mean=2.3, box_cox=0.5)

        self.assertEqual(transf._d, 3)
        self.assertEqual(transf._mean, 2.3)
        self.assertEqual(transf._box_cox, 0.5)


if __name__ == '__main__':
    unittest.main()
