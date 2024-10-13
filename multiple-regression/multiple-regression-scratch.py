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
  
  def optimizemodelparameters(self, features, groundtruth, predictions):
    dW, db = self.gradientdescent(features, groundtruth, predictions)
    self.weight[0] += self.rate * -dW[0]
    self.weight[1] += self.rate * -dW[1]
    self.bias += self.rate * -db
    
  def fit(self, X, ytrue, epochs = 10, out = False):    
    history = {'epoch': [], 'loss': []}
    for epoch in range(epochs):
      yhat = self.MLR(X)
      loss = self.loss(ytrue, yhat)
      self.optimizemodelparameters(X, ytrue, yhat)
      if out:
        print('epoch:', epoch, 'loss:', loss)
      history['epoch'].append(epoch)
      history['loss'].append(loss)
    return history