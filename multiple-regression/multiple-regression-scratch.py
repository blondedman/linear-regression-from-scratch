import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import train_test_split

# loading dataset
sales = pd.read_csv("multiple-regression\sales.csv")

print(sales.head())
print(sales.shape)

# visualizing data
plt.figure(figsize = (15,5))
plt.subplot(3,3,1)
sns.regplot(x = 'TV', y ='sales', data = sales, marker = 'x', color = 'lightblue')
plt.subplot(3,3,2)
sns.regplot(x = 'radio', y ='sales', data = sales, marker = 'x', color = 'lightblue')
plt.subplot(3,3,3)
sns.regplot(x = 'newspaper', y ='sales', data = sales, marker = 'x', color = 'lightblue')

plt.show()


def weightplusbias(w, b):
  weights = np.random.rand(w)
  bias = 0.01 * np.random.rand(b)
  return weights, bias

class multipleregression:
  # gotta work on this // currently incomplete
  def __init__(self, rate = 0.001, w = 3, b = 0):
    self.rate = rate
    self.w = w
    self.b = b
    
  def MLR(self, features, w, b):
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
  
  def modelcoef(self):
    return self.weight, self.bias
  
  
X = sales[['TV','radio','newspaper']]
y = sales['sales']

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = multipleregression()

history = model.fit(X_train, y_train, epochs = 1000)

def learningcurve(model, history):
  coef, bias = model.getmodelcoef()
  plt.figure(figsize = (8,5))
  plt.plot(history['loss']);
  plt.title(f'learning curve # learned weights:{coef} and bias:{bias :.2f}')
  plt.xlabel('epochs')
  plt.ylabel('mean squared error')
  plt.show()

learningcurve(model, history)