from typing import *
import numpy as np
import pandas

import utils
from model import Model
from gradient_descent import gradient_descent

class LogisticModel(Model):
  """
  The logistic model y(x) = sigmoid(θ . x). y(x) is probabilities, not classifications.
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
    (m, _) = X.shape
    X = np.c_[np.ones(m), X]

    return utils.sigmoid(X @ self.θ)

  def __repr__(self) -> str:
    """
    Produce a string like "sigmoid(2.345 + 18.607*x_1 + 13.232*x_2)"
    """

    theta_values = list(self.θ[:, -1])

    first_term = [str(theta_values[0])]
    remaining_terms = [f"{θ_i}*x_{i+1}" for i, θ_i in enumerate(theta_values[1:])]
    inner_string = " + ".join(first_term + remaining_terms)

    return f"sigmoid({inner_string})"

"""
J is our loss function: J(h_theta) = (1/m) sum_{i}^m (-y_i * log(h_theta(x_i)) - (1 - y_i) * log(1 - h_theta(x_i)))
"""

def J(data: np.ndarray, model: Model):
  y = data[:, -1:] # the output is the last column in the data array
  X = data[:, :-1] # everything else is the input
  predicted = model.predict(X)
  return np.mean(y * -np.log(predicted) + (1 - y) * -np.log(1 - predicted))

"""
The derivative of J with respect to θ_i.
"""

def dJ_dθ_i(i: int, data: np.ndarray, model: Model):
  y = data[:, -1:]
  X = data[:, :-1]
  error = model.predict(X) - y

  if i == 0:
    return np.mean(error)

  ith_feature_column = X[:, [i - 1]] # theta_1 is the coefficient on the zeroth column of X
  return np.mean(ith_feature_column * error)

def logistic_regression(data) -> Model:
  return gradient_descent(data, LogisticModel, J, dJ_dθ_i)

def percentage_correct(model, data) -> int:
  """
  Returns a number out of 100.
  """

  y = data[:, -1:]
  X = data[:, :-1]

  predicted = np.round(model.predict(X))
  number_correct = np.sum(predicted == y)
  return int((number_correct / predicted.size) * 100)

if __name__ == "__main__":
  df = pandas.read_csv('boston_housing.csv')
  data = df[['medv', 'chas']].to_numpy()

  model = logistic_regression(data)
  print("Our model's percentage_correct", percentage_correct(model, data))

  # Compare against sklearn.
  import sklearn.linear_model
  y = data[:, -1:].ravel() # sklearn requires input as a 1-d array for some reason.
  X = data[:, :-1]
  sklearn_model = sklearn.linear_model.LogisticRegression().fit(X, y)
  print("sklearn model's percentage_correct", int(sklearn_model.score(X, y) * 100))
