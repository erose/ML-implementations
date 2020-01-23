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
    probabilities = self.feedforward(X)
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
  m, _ = data.shape

  model = cast(NeuralNetwork, model)
  trace = model.feedforward_trace(X)
  Zs = [Z for (Z, _) in trace]
  As = [A for (_, A) in trace]

  # The one-hot vectors indicate which class is accurate. They are useful for vectorizing the cost
  # computation below.
  one_hots = utils.one_hots(y[:, 0]) # one_hots accepts a one-dimensional argument.
  layers = len(trace)
  result = []

  # Compute the gradient for the last layer first. It's a special case.
  δ = As[layers - 1] - one_hots
  grad = _compute_gradient(δ, As[layers - 2])
  result.append(grad)

  # Now, starting with the second-to-last layer, compute each gradient.
  for l in range(layers - 2, 0, -1):
    δ = _compute_delta(δ, model.Θs[l], Zs[l])
    grad = _compute_gradient(δ, As[l - 1])
    result.append(grad)

  # Now that all results are collected, make it an np.ndarray, because only now do we know the shape
  # of such an array.
  result = np.array(result)
  # Reverse the results since the first one is the gradient for the final layer.
  result = result[::-1]
  result /= m # Account for the (1/m) in the loss function.
  return result

def _compute_delta(δ, Θ, Z):
  """
  Based on the following equation.
  δ_l = ((Θ_{l+1})^T @ δ_{l+1}) * sigmoid_gradient(z_l)

  In the equation, δ and Z are vectors, but in the code they are matrices for better performance.
  """

  Θ = Θ[1:] # Chop off the bias weights.

  # Since δ and Θ.T are both matrices, flip the multiplication around so Θ.T acts on the rows of δ.
  return (δ @ Θ.T) * utils.sigmoid_gradient(Z)

def _compute_gradient(δ, A):
  """
  Based on the following equation.
  grad_l = δ_{l+1} @ a_l^T

  In the equation, δ and a are vectors, but in the code they are matrices for better performance.
  """

  A = utils.prepend_column_of_ones(A) # Add bias column in.

  # Since δ and A.T are both matrices, flip the multiplication around so A.T acts on the rows of δ.
  return A.T @ δ

def train_neural_network(data: np.ndarray, nodes_per_layer: List[int], **kwargs) -> NeuralNetwork:
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
  return gradient_descent(data, NeuralNetwork, J, grad_J, initial_Θs, **kwargs)

if __name__ == "__main__":
  pass
