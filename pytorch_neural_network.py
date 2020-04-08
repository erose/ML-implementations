from typing import *
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.nn.modules.module

# TODO: What's the name of this function?
class MyLoss(torch.nn.modules.module.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input, target):
    n, num_classes = input.size()

    # Make one-hots.
    one_hots = torch.zeros(n, num_classes)
    one_hots[range(n), target] = 1 # If target is [t0, t1 ...] this is like assigning one_hots[0][t0] = 1, one_hots[1][t1] = 1, etc.
    return torch.sum(one_hots * -torch.log(input) - (1 - one_hots) * torch.log(1 - input)) / n

def train_and_test(training_data: np.ndarray, test_data: np.ndarray, *, epochs=None, learning_rate=None) -> Tuple[int, int]:
  training_X = torch.from_numpy(training_data[:, :-1]).float()
  training_y = torch.from_numpy(training_data[:, -1]).long()
  test_X = torch.from_numpy(test_data[:, :-1]).float()
  test_y = torch.from_numpy(test_data[:, -1]).long()

  # Only use this architecture, currently.
  model = torch.nn.Sequential(
    torch.nn.Linear(400, 30),
    torch.nn.Sigmoid(),
    torch.nn.Linear(30, 10),
    torch.nn.Sigmoid(),
  )

  criterion = MyLoss()

  # We are really using vanilla (batch) gradient descent, even though we use the SGD optimizer here.
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  for i in range(epochs):
    optimizer.zero_grad()

    output = model(training_X)
    loss = criterion(output, training_y)
    loss.backward()

    if i % 100 == 0:
      print(f"Iteration: {i}")
      print(f"Loss: {loss.item()}")

    optimizer.step()

  output = model(test_X)
  _, predicted = torch.max(output, 1)

  number_correct = (predicted == test_y).sum().item()
  percentage_correct = (number_correct / len(predicted)) * 100
  loss = criterion(output, test_y)

  return (percentage_correct, loss.item())
