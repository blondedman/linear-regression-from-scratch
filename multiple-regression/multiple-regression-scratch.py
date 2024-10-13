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
    return np.mean(np.square((groundtruth - predictions)))
  
  def gradientdescent(self, features, groundtruth, predictions):
    error = groundtruth - predictions
    dW = []
    for column in features.columns:
      dW.append(-2 * np.mean((error * features[column])))
    db = -2 * np.mean(error)
    return dW, db
  
  def optimizemodelparametes(self, features, groundtruth, predictions):
    dW, db = self.gradientdescent(features, groundtruth, predictions)
    self.weight[0] += self.rate * -dW[0]
    self.weight[1] += self.rate * -dW[1]
    self.bias += self.rate * -db