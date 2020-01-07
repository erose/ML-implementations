import unittest
import numpy
from linear_regression_via_gradient_descent import linear_regression

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression_can_learn_doubling(self):
        model = linear_regression(
          numpy.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        )
        self.assertAlmostEqual(model.theta_0, 0.0, places=3)
        self.assertAlmostEqual(model.theta_1, 2.0, places=3)

if __name__ == '__main__':
    unittest.main()
