from typing import *
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
  """
  Abstract class representing a model that makes a prediction given some data.
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

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def __repr__(self) -> str:
    pass
