from typing import *
import numpy as np

from model import Model

def gradient_descent(
  data: np.ndarray,
  model_class: Type[Model],
  J: Callable[[np.ndarray, Model], float],
  grad_J: Callable[[np.ndarray, Model], float],
  initial_parameters: Any,
  iterations=5000
):
  """
  data is an m x n float array.
  """
  m, n = data.shape

  alpha = 0.01 # Arbitrary. Not sure what the principled way to choose this is.  
  model = model_class(initial_parameters)
  
  # Do gradient descent.
  for i in range(iterations):
    gradient = grad_J(data, model)

    model.adjust_by(-alpha * gradient)
    # print(f"Iteration: {i}")
    # print(f"J: {J(data, model)}")
    # print("magnitude of gradient", np.linalg.norm(gradient))
    # print(f"model: {model}")

  return model
