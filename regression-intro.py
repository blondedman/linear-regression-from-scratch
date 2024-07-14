from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import quandl
import math

df = quandl.get('WIKI/GOOGL')
print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 
df['PCT_change'] = (df['Adj. Close'] -df ['Adj. Open']) / df ['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(['label'],axis = 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

# X = X[:-forecast_out+1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = svm.SVR(kernel='poly')
# clf = LinearRegression(n_jobs=10)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)