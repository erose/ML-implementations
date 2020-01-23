import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
  return sigmoid(z) * (1 - sigmoid(z))

def one_hots(a: np.ndarray) -> np.ndarray:
  if a.ndim != 1:
    raise ValueError("Input must be a one-dimensional array.")
  if not is_integer(a):
    raise ValueError("Input must be an array of integers.")

  a = a.astype(int)
  n = np.max(a) + 1
  # Construct the identity matrix, whose rows are the one-hot vectors, then index into it to grab
  # the appropriate ones.
  return np.eye(n)[a]

def is_integer(a: np.ndarray) -> bool:
  """
  Returns true if every object in the array is an integer (i.e. if it is equal to 0 mod 1). Says yes
  for floats that are integers like 4.0.
  """

  return np.all(np.equal(np.mod(a, 1), 0))

def prepend_column_of_ones(A: np.ndarray) -> np.ndarray:
  """
  Adds a column of ones to the front of A.
  """

  (m, _) = A.shape
  return np.c_[np.ones(m), A]

def prepend_row_of_ones(A: np.ndarray) -> np.ndarray:
  """
  Adds a row of ones to the top of A.
  """

  (_, n) = A.shape
  return np.r_[[np.ones(n)], A]
