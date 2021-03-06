from typing import *
import numpy as np
import pandas

import utils
from model import Model
from gradient_descent import gradient_descent

class LinearModel(Model):
  """
  The linear model y(x) = θ . x
  """

  def __init__(self, θ: np.ndarray):
    """
    θ is an array of floats θ_0, θ_1 ...
    """

    if θ.ndim != 2:
      raise ValueError(f"Expected θ to be a column vector, but it had ndim == {θ.ndim}")
    if θ.shape[1] != 1:
      raise ValueError(f"Expected θ to be a column vector, but it had shape == {θ.shape}")

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
    X = utils.prepend_column_of_ones(X)
    return X @ self.θ

  def __repr__(self) -> str:
    """
    Produce a string like "2.345 + 18.607*x_1 + 13.232*x_2"
    """

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

def grad_J(data: np.ndarray, model: Model):
  # Note this function happens to be the same as grad_J in logistic_regression.py.
  y = data[:, -1:]
  X = data[:, :-1]
  _, n = data.shape

  error = model.predict(X) - y

  result = np.zeros((n, 1))
  result[0] = np.mean(error)
  for i in range(1, n):
    ith_feature_column = X[:, [i - 1]] # theta_1 is the coefficient on the zeroth column of X
    result[i] = np.mean(ith_feature_column * error)

  return result

def linear_regression(data) -> Model:
  m, n = data.shape
  initial_parameters = np.zeros((n, 1))

  return gradient_descent(data, LinearModel, J, grad_J, initial_parameters)

if __name__ == "__main__":
  df = pandas.read_csv('data/boston_housing.csv')
  data = df[['dis', 'rm', 'medv']].to_numpy()

  model = linear_regression(data)
  print("Our model", model)

  # Compare against sklearn.
  import sklearn.linear_model
  y = data[:, -1:]
  X = data[:, :-1]
  sklearn_model = sklearn.linear_model.LinearRegression().fit(X, y)
  print("sklearn's model", sklearn_model.intercept_, sklearn_model.coef_)
