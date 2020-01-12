from typing import *
import numpy as np

from model import Model

def gradient_descent(
  data: np.ndarray, model_class: Type[Model],
  J: Callable[[np.ndarray, Model], float],
  dJ_dθ_i: Callable[[int, np.ndarray, Model], float],
  iterations=5000
):
  """
  data is an m x n float array.
  """
  m, n = data.shape

  alpha = 0.01 # Arbitrary. Not sure what the principled way to choose this is.
  initial_parameters = np.zeros((n, 1)) # Arbitrary initial guess.
  epsilon = 0.001 # Arbitrary.
  
  model = model_class(initial_parameters)
  
  # Do gradient descent.
  for i in range(iterations):
    # Gather the partial derivatives with respect to each θ_i into the gradient.
    gradient = np.array([[dJ_dθ_i(i, data, model)] for i in range(n)])

    model.adjust_by(-alpha * gradient)
    # print(f"Iteration: {i}")
    # print(f"J: {J(data, model)}")
    # print("magnitude of gradient", np.linalg.norm(gradient))
    # print(f"model: {model}")

  return model
