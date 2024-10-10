import numpy as np

def weightplusbias(w, b):
  weights = np.random.rand(w)
  bias = 0.01 * np.random.rand(b)
  return weights, bias

def MLR(features, weights, bias):
  return (features@weights) + bias