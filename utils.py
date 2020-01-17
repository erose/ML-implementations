import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-z))

def prepend_column_of_ones(A: np.ndarray) -> np.ndarray:
  """
  Adds a column of ones to the front of A.
  """

  (m, _) = A.shape
  return np.c_[np.ones(m), A]
