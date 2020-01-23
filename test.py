import unittest
import copy
import numpy as np
import numpy.testing

import utils
import linear_regression as linr
import logistic_regression as logr
import neural_network as nn

class TestUtils(unittest.TestCase):
  def test_one_hots(self):
    a = np.array([6, 0, 3])

    numpy.testing.assert_array_equal(
      utils.one_hots(a),
      np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
      ])
    )

class TestLinearRegression(unittest.TestCase):
  def test_model_can_be_printed(self):
    model = linr.LinearModel(np.array([[2, 1, 0]]).T)
    self.assertEqual(str(model), "2 + 1*x_1 + 0*x_2")

  def test_cost_function_on_constant_model(self):
    dumb_model = linr.LinearModel(np.array([[6, 0]]).T)

    X = np.array([
      [1, 2], # cost (6 - 2)^2 = 16
    ])
    cost = linr.J(X, dumb_model)
    self.assertEqual(cost, 16)

    X = np.array([
      [1, 2], # cost (6 - 2)^2 = 16
      [0, 0], # cost (6 - 0)^2 = 36
    ])
    cost = linr.J(X, dumb_model)
    self.assertEqual(cost, (16 + 36) / 2)

  def test_cost_function(self):
    X = np.array([
      [1, 2], # cost (6 + 1*1 - 2)^2 = 25
    ])
    model = linr.LinearModel(np.array([[6, 1]]).T)
    cost = linr.J(X, model)
    self.assertEqual(cost, 25)

    # now test a correct model that should have zero cost
    X = np.array([
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ])
    correct_model = linr.LinearModel(np.array([[0, 2]]).T)
    cost = linr.J(X, correct_model)
    self.assertEqual(cost, 0)

  def test_linear_regression_can_learn_doubling(self):
    model = linr.linear_regression(
      np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0]
      ]),
    )

    prediction = model.predict(np.array([[6.0]]))
    self.assertAlmostEqual(prediction[0][0], 12.0, places=3)

class TestLogisticRegression(unittest.TestCase):
  def test_cost_function_on_constant_model(self):
    dumb_model = logr.LogisticModel(np.array([[6, 0]]).T)
    X = np.array([
      [1, 1], # cost -log(sigmoid(6)) ~= 0.02 
    ])
    cost = logr.J(X, dumb_model)
    self.assertAlmostEqual(cost, 0.002, places=3)

    X = np.array([
      [1, 0], # cost -log(1 - sigmoid(6)) ~= 6.002 
    ])
    cost = logr.J(X, dumb_model)
    self.assertAlmostEqual(cost, 6.002, places=3)

  def test_logistic_regression_can_learn_cutoff(self):
    model = logr.logistic_regression(
      np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [5.0, 1.0]
      ]),
    )

    prediction = model.predict(np.array([[5.0]]))
    self.assertAlmostEqual(prediction[0][0], 0.988, places=3)

class TestNeuralNetwork(unittest.TestCase):
  def test_can_predict_with_simple_architecture(self):
    # The network has two layers, with two nodes in the first and one node in the second. It's just
    # a logistic regressor that takes two inputs.
    model = nn.NeuralNetwork([
      np.array([
        [0.0],
        [1.0],
        [1.0]
      ]),
    ])

    X = np.array([
      [2.0, 0.0],
      [0.0, -1.0],
    ])
    expected_output = np.array([
      [utils.sigmoid(2)],
      [utils.sigmoid(-1)]
    ])

    numpy.testing.assert_almost_equal(model.feedforward(X), expected_output, decimal=3)

  def test_cost_function_on_simple_model(self):
    # The network has two layers, with two input nodes and two output nodes. Both output nodes
    # compute the same function.
    model = nn.NeuralNetwork([
      np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0]
      ]),
    ])

    data = np.array([
      [2.0,  0.0, 0],
      [0.0, -1.0, 1],
    ])

    log, σ = np.log, utils.sigmoid
    expected_cost = np.mean([
      # We always predict 0, so the first example suffers:
      #  - a cost for being underconfident in the first output node.
      #  - a cost for being overconfident in the second output node.
      -log(σ(2)) + -log(1 - σ(2)),

      # We always predict 0, so the second example suffers:
      #  - a cost for being overconfident in the first output node.
      #  - a cost for being underconfident in the second output node.
      -log(1 - σ(-1)) + -log(σ(-1)),
    ])
    cost = nn.J(data, model)

    self.assertAlmostEqual(expected_cost, cost, places=3)

  def test_can_compute_gradient_of_cost_function(self):
    # The network has three layers, with two input nodes and two output nodes per layer.
    model = nn.NeuralNetwork([
      np.array([
        [0.0,  0.0],
        [1.0,  0.5],
        [1.0, -1.0]
      ]),

      np.array([
        [ 0.1, 0.0],
        [-8.0, 1.0],
        [ 0.0, 1.0]
      ]),
    ])

    data = np.array([
      [2.0,  0.0, 0],
      [0.0, -1.0, 1],
    ])

    grad = nn.grad_J(data, model)
    approx_grad = self.compute_approximate_gradient_by_finite_difference(data, model)

    numpy.testing.assert_almost_equal(grad, approx_grad, decimal=3)

  def compute_approximate_gradient_by_finite_difference(self, data: np.ndarray, net: nn.NeuralNetwork) -> np.ndarray:
    """
    Utility method only used in test. This is a "brute force" way of computing the gradient we check
    against our more efficient, but trickier, implementation.
    """
    ϵ = 0.001

    result = []
    for l, Θ in enumerate(net.Θs):
      m, n = Θ.shape
      dΘ = np.zeros((m, n))
      for i in range(m):
        for j in range(n):
          # We want to compute d/dΘ_ij numerically. What we do is vary Θ_ij slightly, compute the
          # cost before and after, and look at the difference between them.

          # In math, we are doing d/dΘ_ij = (J(..., Θ_ij + ϵ, ...) - J(..., Θ_ij, ...)) / ϵ.

          # Our cost function doesn't take a list of theta matrices directly, so we construct a
          # plus_epsilon model in order to calcuate the above. We do this
          # with a lot of copying. Happily this is just a test method, so hopefully the inefficiency
          # won't bite us too badly.

          Θs_copy = copy.deepcopy(net.Θs)
          Θs_copy[l][i][j] += ϵ
          net_plus_epsilon = nn.NeuralNetwork(Θs_copy)

          dΘ[i][j] = (nn.J(data, net_plus_epsilon) - nn.J(data, net)) / ϵ

      result.append(dΘ)

    return np.array(result)

  def test_cost_decreases_when_gradient_is_applied(self):
    # The network has two layers, with two input nodes and two output nodes. Both output nodes
    # compute the same function.
    model = nn.NeuralNetwork([
      np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0]
      ]),
    ])

    data = np.array([
      [2.0,  0.0, 0],
      [0.0, -1.0, 1],
    ])
    gradient = self.compute_approximate_gradient_by_finite_difference(data, model)
    cost_before = nn.J(data, model)

    alpha = 0.01 # Arbitrary.
    model.adjust_by(-alpha * gradient)

    cost_after = nn.J(data, model)
    self.assertTrue(cost_after < cost_before)

  def test_can_learn_left_vs_right(self):
    data = np.array([
      [ 2.0,  0.0, 0],
      [ 0.0, -1.0, 1],
      [ 0.0, -6.0, 1],
      [-1.0,  0.0, 0],
    ])
    # Training uses random initialization, so seed the generator to ensure we get the same value
    # every time.
    np.random.seed(0)
    model = nn.train_neural_network(data, [2, 2, 2])

    X = np.array([
      [2.0,  0.0],
      [0.0, -1.0],
    ])
    prediction = model.feedforward(X)

    expected_prediction = np.array([
      [0.898, 0.102],
      [0.265, 0.733],
    ])
    numpy.testing.assert_almost_equal(prediction, expected_prediction, decimal=3)

if __name__ == '__main__':
  unittest.main()
