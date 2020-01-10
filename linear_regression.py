from typing import *
import numpy as np
import pandas

class Model:
  """
  The linear model y(x) = θ . x
  """

  def __init__(self, θ: np.ndarray):
    """
    θ is an array of floats θ_0, θ_1 ... such that y(x) = θ_0 + x_1*θ_1 + ...
    """

    if θ.ndim != 2:
      raise ValueError(f"Expected model input to be a column vector, but it had ndim == {θ.ndim}")

    self.θ = θ

  def adjust_by(self, delta: np.ndarray):
    """
    delta is an array of floats ordered the same way as θ.
    """

    self.θ += delta

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    X is an m x n array of feature vectors, where n+1 is the number of elements in θ. Theta has
    one more element because in addition to a coefficient for each feature, it has a constant
    coefficient θ_0.
    """

    # Prepend a column of ones to X; these will be dotted with θ_0.
    (m, _) = X.shape
    X = np.c_[np.ones(m), X]

    return X @ self.θ

  def __repr__(self):
    theta_values = list(self.θ[:, -1])

    first_term = [str(theta_values[0])]
    remaining_terms = [f"{θ_i}*x_{i+1}" for i, θ_i in enumerate(theta_values[1:])]
    return " + ".join(first_term + remaining_terms)

"""
J is our loss function: J(h_theta) = (1/2m) sum_{i}^m (h_theta(x_i) - y_i)^2
"""

def J(data: np.ndarray, model: Model):
  y = data[:, -1:] # the output is the last column in the data array
  X = data[:, :-1] # everything else is the input
  error = model.predict(X) - y
  return np.mean(error ** 2)

def d_J_d_theta_0(data: np.ndarray, model: Model):
  y = data[:, -1:]
  X = data[:, :-1]
  error = model.predict(X) - y
  return np.mean(error)

def d_J_d_theta_i(i: int, data: np.ndarray, model: Model):
  y = data[:, -1:]
  X = data[:, :-1]
  error = model.predict(X) - y

  ith_feature_column = X[:, [i - 1]] # theta_1 is the coefficient on the zeroth column of X
  return np.mean(ith_feature_column * error)

def linear_regression(data: np.ndarray) -> Model:
  """
  data is an m x n float array.
  """
  m, n = data.shape

  alpha = 0.1 # Arbitrary. Not sure what the principled way to choose this is.
  initial_parameters = np.array([[0.0], [0.0]]) # Arbitrary initial guess.
  epsilon = 0.00001 # Arbitrary.
  
  model = Model(initial_parameters)
  
  # Do gradient descent until convergence.
  i = 0
  while True:
    # Gather the partial derivatives with respect to each theta_i into the gradient.
    gradient = np.array([
      [d_J_d_theta_0(data, model)],
      *[[d_J_d_theta_i(i, data, model)] for i in range(1, n)],
    ])

    if np.linalg.norm(gradient) < epsilon:
      break

    model.adjust_by(-alpha * gradient)
    # print(f"Iteration: {i}")
    # print(f"J: {J(data, model)}")
    i += 1

  return model

if __name__ == "__main__":
  df = pandas.read_csv('boston_housing.csv')
  distance_to_median_value = df[['dis', 'medv']].to_numpy()

  model = linear_regression(distance_to_median_value)
  print(model)
