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
  def __repr__(self) -> str:
    pass
