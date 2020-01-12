from typing import *
import numpy as np
import pandas

from model import Model

class LogisticModel(Model):
  """
  The logistic model y(x) = sigmoid(θ . x). y(x) is probabilities, not classifications.
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

    return self.sigmoid(X @ self.θ)

  def sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-z))

  def __repr__(self) -> str:
    pass
