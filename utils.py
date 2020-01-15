import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-z))
