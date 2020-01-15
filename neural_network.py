from typing import *
import numpy as np
import pandas

import utils
from model import Model

class NeuralNetwork(Model):
  """
  A neural network with n layers.
  """

  def __init__(self, Θs: List[np.ndarray]):
    """
    Θs is a list of n-1 matrices, where n is the number of layers in the network. Θs[i] is the
    matrix describing the connections between layer i and layer i+1, such that Θs[i][j][k] is the
    weight of the connection between node j of layer i and node k of layer i+1.

    Each Θs[i] may be of a different dimensionality, with the restriction that if Θs[i] is p x q,
    Θs[i+1] is q+1 x r. In other words, we need to be able to feed the output of one layer into the
    input of the next layer. (The +1 is because we add an entry to account for the always-1 "bias"
    node.)
    """

    for i, (Θa, Θb) in enumerate(zip(Θs, Θs[1:])):
      (p, q) = Θa.shape
      (r, s) = Θb.shape
      if q != r+1:
        raise ValueError(f"Expected matrices {i} and {i+1} in Θs to be compatible, but matrix {i} had shape {Θa.shape} while matrix {i+1} had shape {Θb.shape}.")

    self.Θs = Θs

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    X is an a x b array of feature vectors, where a+1 is the number of nodes in the first layer of the
    network. (The +1 is because of the bias node.)
    """

    # For each step, we add in the ones for the bias node, compute Z, then compute A.
    Z = X
    A = Z

    for i in range(len(self.Θs)):
      A = utils.prepend_ones(A)
      Z = A @ self.Θs[i]
      A = utils.sigmoid(Z)

    h_theta = A
    return h_theta

  def __repr__(self) -> str:
    raise Exception("TODO")
