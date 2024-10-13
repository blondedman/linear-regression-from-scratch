import numpy as np

def weightplusbias(w, b):
  weights = np.random.rand(w)
  bias = 0.01 * np.random.rand(b)
  return weights, bias

class multipleregression:
  
  def __init__(self, rate, w, b):
    self.rate = rate
    self.w = w
    self.b = b
    
  def MLR(self, features, weights, bias):
    return (features@self.w) + self.b

  def loss(self, groundtruth, predictions):
    return np.mean(np.square((groundtruth-predictions)))
  
  def gradientdescent(self, features, groundtruth, predictions):
    pass