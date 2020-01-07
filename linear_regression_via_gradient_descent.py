from typing import *
import numpy as np
import pandas

class Model:
  """
  The linear model y(x) = theta_0 + x*theta_1
  """

  def __init__(self, theta_0: float, theta_1: float):
    self.theta_0 = theta_0
    self.theta_1 = theta_1

  def adjust_by(self, delta: np.ndarray):
    """
    delta is a two-element array.
    """

    self.theta_0 += delta[0]
    self.theta_1 += delta[1]

  def predict(self, x: float) -> float:
    return x*self.theta_1 + self.theta_0

  def __repr__(self):
    return f"x*{self.theta_1} + {self.theta_0}"

"""
J is our loss function: J(h_theta) = (1/2m) sum_{i}^m (h_theta(x_i) - y_i)^2
"""

def d_J_d_theta_0(data: np.ndarray, model: Model):
  error = np.array([model.predict(x) - y for x, y in data])
  return np.mean(error)

def d_J_d_theta_1(data: np.ndarray, model: Model):
  error = np.array([x*(model.predict(x) - y) for x, y in data])
  return np.mean(error)

def linear_regression(data: np.ndarray) -> Model:
  """
  data is an array of two-element arrays containing floats.
  """

  alpha = 0.1 # Arbitrary. Not sure what the principled way to choose this is.
  initial_parameters = (1.0, 1.0) # Arbitrary initial guess.
  epsilon = 0.00001 # Arbitrary.
  
  model = Model(*initial_parameters)
  
  # Do gradient descent until convergence.
  while True:
    delta = np.array([d_J_d_theta_0(data, model), d_J_d_theta_1(data, model)])

    if np.linalg.norm(delta) < epsilon:
      break

    model.adjust_by(-1 * alpha * delta)

  return model

if __name__ == "__main__":
  df = pandas.read_csv('boston_housing.csv')
  distance_to_median_value = df[['dis', 'medv']].to_numpy()

  model = linear_regression(distance_to_median_value)
  print(model)
