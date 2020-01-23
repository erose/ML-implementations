import copy

import neural_network as nn
import numpy as np

def compute_approximate_gradient_by_finite_difference(data: np.ndarray, net: nn.NeuralNetwork) -> np.ndarray:
  """
  This is a "brute force" way of computing the gradient we check against our more efficient, but
  trickier, implementation.
  """
  ϵ = 0.001

  result = []
  for l, Θ in enumerate(net.Θs):
    m, n = Θ.shape
    dΘ = np.zeros((m, n))
    for i in range(m):
      for j in range(n):
        # We want to compute d/dΘ_ij numerically. What we do is vary Θ_ij slightly, compute the
        # cost before and after, and look at the difference between them.

        # In math, we are doing d/dΘ_ij = (J(..., Θ_ij + ϵ, ...) - J(..., Θ_ij, ...)) / ϵ.

        # Our cost function doesn't take a list of theta matrices directly, so we construct a
        # plus_epsilon model in order to calcuate the above. We do this
        # with a lot of copying. Happily this is just a test method, so hopefully the inefficiency
        # won't bite us too badly.

        Θs_copy = copy.deepcopy(net.Θs)
        Θs_copy[l][i][j] += ϵ
        net_plus_epsilon = nn.NeuralNetwork(Θs_copy)

        dΘ[i][j] = (nn.J(data, net_plus_epsilon) - nn.J(data, net)) / ϵ

    result.append(dΘ)

  return np.array(result)
