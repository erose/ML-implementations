from typing import *
import numpy as np
import scipy.io

import utils
from gradient_descent import gradient_descent
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
      if q+1 != r:
        raise ValueError(f"Expected matrices {i} and {i+1} in Θs to be compatible, but matrix {i} had shape {Θa.shape} while matrix {i+1} had shape {Θb.shape}.")

    self.Θs = Θs

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    X is an a x b array of feature vectors, where b+1 is the number of nodes in the first layer of the
    network. (The +1 is because of the bias node.)
    """

    # For each step, we add in the ones for the bias node, compute the weighted input for this layer
    # Z, then the activation for this layer A.
    Z = X
    A = Z

    for i in range(len(self.Θs)):
      # To every example, add in a 1 as a constant to be multiplied by the bias term in self.Θs[i].
      A = utils.prepend_column_of_ones(A)
      
      Z = A @ self.Θs[i] # self.Θs[i] acts on the rows of A.
      A = utils.sigmoid(Z)

    h_theta = A
    return h_theta

  def __repr__(self) -> str:
    raise NotImplemented

"""
J is our loss function.
"""

def J(data: np.ndarray, model: Model):
  y = data[:, -1:] # the output is the last column in the data array
  X = data[:, :-1] # everything else is the input
  m, n = X.shape

  h_theta = model.predict(X)
  predictions = np.argmax(h_theta, axis=1)

  result = 0
  # Nonregularized term.
  for i in range(m):
    for output in h_theta[i]:
      if predictions[i] == y[i]:
        result += -np.log(output)
      else:
        result += -np.log(1 - output)

  result /= m
  return result

def train_neural_network(data: np.ndarray, layer_shapes: List[Tuple[int, int]]) -> Model:
  # Initialize our weights to random small values.
  epsilon = 0.1

  initial_Θs = [np.random.uniform(low=-epsilon, high=epsilon, size=shape) for shape in layer_shapes]
  return gradient_descent(data, NeuralNetwork, J, dJ_dθ_i, initial_Θs)

if __name__ == "__main__":
  data_mat = scipy.io.loadmat('mnist_data.mat')
  X = data_mat['X']
  y = data_mat['y'] % 10 # The data encodes '0' as '10'.

  weights_mat = scipy.io.loadmat('mnist_weights.mat')
  # Take the transpose because the convention used in these files is different than ours.
  theta1 = weights_mat['Theta1'].T
  theta2 = weights_mat['Theta2'].T
  model = NeuralNetwork([theta1, theta2])

  probabilities = model.predict(X)
  # Because the weights think of '0' as '10', the last probability in a row is really the
  # probability that the digit is '0', not '9'. So we cyclically shift each row one to the right to
  # correct this.
  probabilities = np.roll(probabilities, 1)
  predictions = np.argmax(probabilities, axis=1)

  print("Training set accuracy:", np.mean(predictions == y[:, 0]))
