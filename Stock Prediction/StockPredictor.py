# Following sentdex's tutorial series on creating a simple linear regression stock predictor
import pandas as pd
import quandl
import math, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

# The stock to use
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
# Calculate volatility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Calculate actual change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#         price           x            x           x
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

# The number multiplying it is percentage out looking to predict
forecast_out = int(math.ceil(0.1*len(df)))
# How many days out is it
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)


# Features are usually X
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]   # No y values so don't train or test on this data
X = X[:-forecast_out]


df.dropna(inplace=True)

# Labels are y
y = np.array(df['label'])

# model_selection is used over cross validation now
# X_train and y_train are used to fit classifiers
# model_selection will shuffle the data to improve results
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

############################################################################################################
# Can comment out this section so it doesn't have to compute training again
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
# Save the classifier here to prevent from having to always retrain
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
############################################################################################################

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
# Here are the guess stock prices for the next 30 days
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.suptitle('WIKI/GOOGL', fontsize=14, fontweight='bold')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
