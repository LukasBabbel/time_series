import unittest
from arma import PureARMA


class Test_PureARMA(unittest.TestCase):
    def test_para_return(self):
        model = PureARMA([1, 2, 3], [4, 5], 2)
        self.assertEqual(model.get_params(), ([1, 2, 3], [4, 5], 2))

    def test_parameters(self):
        model = PureARMA([-1, 2, 3], [4, 5], 2)
        self.assertEqual(model.get_phi(2), 2)
        self.assertEqual(model.get_phi(0), 1)
        self.assertEqual(model.get_phi(4), 0)

        self.assertEqual(model.get_theta(2), 5)
        self.assertEqual(model.get_theta(0), 1)
        self.assertEqual(model.get_theta(3), 0)

    def test_ma_infty(self):
        model_ma = PureARMA([], [1, 2, 3])
        model_ar = PureARMA([0.5])

        self.assertEqual(model_ma.get_ma_infty_coef(0), 1)
        self.assertEqual(model_ma.get_ma_infty_coef(2), 2)
        self.assertEqual(model_ma.get_ma_infty_coef(3), 3)
        self.assertEqual(model_ma.get_ma_infty_coef(5), 0)
        self.assertEqual(model_ma.get_ma_infty_coef(3), 3)

        self.assertEqual(model_ar.get_ma_infty_coef(0), 1)
        self.assertEqual(model_ar.get_ma_infty_coef(1), 0.5)
        self.assertEqual(model_ar.get_ma_infty_coef(2), 0.25)
        self.assertEqual(model_ar.get_ma_infty_coef(50), 0.5 ** 50)

    def test_acf(self):
        model_332 = PureARMA([-1, 0.25], [1], 3)
        model_ar = PureARMA([0.5])
        model_ma = PureARMA([], [1, 2, 3])

        for k in range(100):
            self.assertEqual(model_332.acf(k), 3 * 2 ** -k * (32 / 3 + 8 * k))

        for k in range(100):
            self.assertEqual(model_ar.acf(k), 0.5 ** k)

        for k in range(4, 100):
            self.assertEqual(model_ma.acf(k), 0)

if __name__ == '__main__':
    unittest.main()
