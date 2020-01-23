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

  def adjust_by(self, deltas: List[np.ndarray]):
    """
    delta is a list of arrays of floats, one for each element in Θs, each with the same shape as
    their corresponding matrix.
    """

    for i in range(len(deltas)):
      self.Θs[i] += deltas[i]

  def predict(self, X: np.ndarray) -> np.ndarray:
    probabilities = model.feedforward(X)
    
    # Because the weights think of '0' as '10', the last probability in a row is really the
    # probability that the digit is '0', not '9'. So we cyclically shift each row one to the right to
    # correct this.
    probabilities = np.roll(probabilities, 1)
    
    predictions = np.argmax(probabilities, axis=1)
    return predictions

  def feedforward(self, X: np.ndarray) -> np.ndarray:
    # The activation of the last layer is the probabilities we want.
    trace = self.feedforward_trace(X)
    _, A_last = trace[-1]

    h_theta = A_last
    return h_theta

  def feedforward_trace(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    X is an a x b array of feature vectors, where b+1 is the number of nodes in the first layer of the
    network. (The +1 is because of the bias node.)

    Returns a list of (Z, A) matrices, where Z_i is the input to layer i and A_i is the output of
    layer i.
    """

    # For each step, we add in the ones for the bias node, compute the weighted input for this layer
    # Z, then the activation for this layer A.
    Z = X
    A = Z
    trace = [(Z, A)]

    for i in range(len(self.Θs)):
      # To every example, add in a 1 as a constant to be multiplied by the bias term in self.Θs[i].
      A = utils.prepend_column_of_ones(A)
      
      Z = A @ self.Θs[i] # self.Θs[i] acts on the rows of A.
      A = utils.sigmoid(Z)

      trace.append((Z, A))

    return trace

  def __repr__(self) -> str:
    raise NotImplemented

"""
J is our loss function.
"""

def J(data: np.ndarray, model: Model) -> float:
  y = data[:, -1:] # the output is the last column in the data array
  X = data[:, :-1] # everything else is the input
  m, _ = X.shape

  h_theta = model.feedforward(X)
  # The one-hot vectors indicate which class is accurate. They are useful for vectorizing the cost
  # computation below.
  one_hots = utils.one_hots(y[:, 0]) # one_hots accepts a one-dimensional argument.
  
  # If the prediction is accurate, penalize for distance to probability 1, else penalize for
  # distance from probability 0.
  result = np.sum(one_hots * -np.log(h_theta) - (1 - one_hots) * np.log(1 - h_theta)) / m
  return result

def grad_J(data: np.ndarray, model: Model):
  y = data[:, -1:]
  X = data[:, :-1]
  m, n = data.shape

  model = cast(NeuralNetwork, model)
  trace = model.feedforward_trace(X)
  # The one-hot vectors indicate which class is accurate. They are useful for vectorizing the cost
  # computation below.
  one_hots = utils.one_hots(y[:, 0]) # one_hots accepts a one-dimensional argument.
  result = []

  Zs = [Z for (Z, _) in trace]
  As = [A for (_, A) in trace]

  A = As[len(trace) - 1]
  A_prev = As[len(trace) - 2]
  
  δ = A - one_hots
  A_prev = utils.prepend_column_of_ones(A_prev)
  result.append(δ.T @ A_prev / m)

  for step in range(len(trace) - 2, 0, -1):
    A_prev = As[step - 1]
    Z = Zs[step]
    Θ = model.Θs[step]

    Θ = Θ[1:] # chop off the bias weights
    δ = (δ @ Θ.T) * utils.sigmoid_gradient(Z)
    A_prev = utils.prepend_column_of_ones(A_prev)
    result.append(δ.T @ A_prev / m)

  # Transpose the results and return them in reverse order, as a numpy array.
  return np.array([dΘ.T for dΘ in result[::-1]])

def train_neural_network(data: np.ndarray, nodes_per_layer: List[int]) -> NeuralNetwork:
  # Initialize our weights to random small values.
  epsilon = 0.1

  # Decide the shapes of the adjacency matrices. If the first layer has 2 nodes, the second has 3
  # nodes, and the last has 1 node, then the shapes should be [(2+1, 3), (3+1, 1)]. (The +1 is
  # because of the bias layer.)
  Θ_shapes = list(zip(nodes_per_layer, nodes_per_layer[1:]))
  for i in range(len(Θ_shapes)):
    m, n = Θ_shapes[i]
    Θ_shapes[i] = (m + 1, n)

  initial_Θs = [np.random.uniform(low=-epsilon, high=epsilon, size=shape) for shape in Θ_shapes]
  return gradient_descent(data, NeuralNetwork, J, grad_J, initial_Θs)

if __name__ == "__main__":
  data_mat = scipy.io.loadmat('mnist_data.mat')
  X = data_mat['X']
  y = data_mat['y'] % 10 # The data encodes '0' as '10'.

  weights_mat = scipy.io.loadmat('mnist_weights.mat')
  # Take the transpose because the convention used in these files is different than ours.
  theta1 = weights_mat['Theta1'].T
  theta2 = weights_mat['Theta2'].T

  model = NeuralNetwork([theta1, theta2])
  predictions = model.predict(X)

  print("Training set accuracy:", np.mean(predictions == y[:, 0]))
