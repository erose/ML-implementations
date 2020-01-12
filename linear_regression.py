from typing import *
import numpy as np
import pandas

from model import Model
from gradient_descent import gradient_descent

class LinearModel(Model):
  """
  The linear model y(x) = θ . x
  """

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

  def __repr__(self) -> str:
    """
    produce a string like "2.345 + 18.607*x_1 + 13.232*x_2"
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

def linear_regression(data) -> Model:
  return gradient_descent(data, LinearModel, J, dJ_dθ_i)

if __name__ == "__main__":
  df = pandas.read_csv('boston_housing.csv')
  data = df[['dis', 'rm', 'medv']].to_numpy()

  model = linear_regression(data)
  print(model)