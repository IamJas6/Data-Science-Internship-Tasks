import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

"""### *Load Dataset*"""
#importing data and reading as csv
df=pd.read_csv('wso.csv')

"""### *Summarize Dataset*"""
print(df.head())                                #From the first five rows, we can see the data
print(df.tail())                                #From the last five rows, we can see the data
print(df.shape)                                 #From this, we got to know that there are 3201 rows of data available and for each row, we have 7 different features or columns.
print(df.describe())                            #this method is used to generate descriptive statistics of a Data
print(df.info())                                #this method is used to display a concise summary of the Data
print(df.isnull)                                #used to check null values if present in the data

'''Exploratory Data Analysis'''
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title('watsco stock close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
print(df.head())

'''check for null values'''
print(df.isnull().sum())

'''distibution plot'''
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
    plt.show()

'''adding extra vairables to make model understand'''
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()


'''Data Splitting and Normalization'''
features = df[['open-close', 'low-high']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


'''Train the model'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

'''Make predictions'''
predictions = model.predict(X_valid)

'''Evaluate the model'''
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_valid, predictions)
print(f'Mean Squared Error: {mse}')


'''visualize Acutal close price'''
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Actual Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.show()

'''Visualize the predictions'''
plt.figure(figsize=(12, 6))
plt.plot(X_valid, predictions, label='Predicted Close Price', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Market Prediction')
plt.legend()
plt.show()
