import unittest
import numpy as np
import linear_regression as lr

class TestModel(unittest.TestCase):
  def test_model_can_be_printed(self):
    model = lr.Model(np.array([[2, 1, 0]]).T)
    self.assertEqual(str(model), "2 + 1*x_1 + 0*x_2")

class TestLinearRegression(unittest.TestCase):
  def test_cost_function_on_constant_models(self):
    data = np.array([
      [1, 2], # cost (6 - 2)^2 = 16
    ])
    dumb_model = lr.Model(np.array([[6, 0]]).T)
    cost = lr.J(data, dumb_model)
    self.assertEqual(cost, 16)

    data = np.array([
      [1, 2], # cost (6 - 2)^2 = 16
      [0, 0], # cost (6 - 0)^2 = 36
    ])
    dumb_model = lr.Model(np.array([[6, 0]]).T)
    cost = lr.J(data, dumb_model)
    self.assertEqual(cost, (16 + 36) / 2)

  def test_cost_function(self):
    data = np.array([
      [1, 2], # cost (6 + 1*1 - 2)^2 = 25
    ])
    model = lr.Model(np.array([[6, 1]]).T)
    cost = lr.J(data, model)
    self.assertEqual(cost, 25)

    # now test a correct model that should have zero cost
    data = np.array([
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ])
    correct_model = lr.Model(np.array([[0, 2]]).T)
    cost = lr.J(data, correct_model)
    self.assertEqual(cost, 0)

  def test_linear_regression_can_learn_doubling(self):
    model = lr.linear_regression(
      np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
      iterations=5000
    )
    self.assertAlmostEqual(model.θ[0][0], 0.0, places=3)
    self.assertAlmostEqual(model.θ[1][0], 2.0, places=3)

if __name__ == '__main__':
  unittest.main()
