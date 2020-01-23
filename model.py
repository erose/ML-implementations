from typing import *
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
  """
  Abstract class representing a model that makes a prediction given some data.
  """

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def adjust_by(self, delta: Any):
    pass

  @abstractmethod
  def __repr__(self) -> str:
    pass
