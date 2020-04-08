from typing import *
import numpy as np

from model import Model

T = TypeVar('T', bound=Model)
def gradient_descent(
  data: np.ndarray,
  model_class: Type[T],
  J: Callable[[np.ndarray, T], float],
  grad_J: Callable[[np.ndarray, T], float],
  initial_parameters: Any,
  learning_rate: float = 0.01,
  verbose: bool = False,
  epochs=5000
):
  """
  data is an m x n float array.
  """
  m, n = data.shape

  model = model_class(initial_parameters)
  
  # Do gradient descent.
  for i in range(epochs):
    gradient = grad_J(data, model)

    model.adjust_by(-learning_rate * gradient)
    if verbose and i % 100 == 0:
      print(f"Iteration: {i}")
      print(f"J: {J(data, model)}")
      # print("magnitude of gradient", np.linalg.norm(gradient))
      # print(f"model: {model}")

  return model
