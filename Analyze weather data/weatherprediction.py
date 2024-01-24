import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the weather data from a CSV file
weather_data = pd.read_csv('losvegas_weather_data.csv')


print(weather_data.head())
print(weather_data.describe())
print(weather_data.info())
print(weather_data.isna().sum())

#cleaning Nan and filling
cleaned=weather_data.dropna(subset=['humidity'])
weather_data['humidity'].fillna(weather_data['humidity'].mean(),inplace=True)
print(weather_data['humidity'])
print(weather_data['humidity'].isna().sum())

#maping string to float
weather_data['weather_description']=weather_data['weather_description'].map({'sky is clear':0,'few clouds':1,'scattered clouds':2,'light rain':3,'broken clouds':4,'overcast clouds':5,'proximity thunderstorm':6,'mist':7,'haze':8,'heavy intensity rain':9})
print(weather_data['weather_description'].head())

#cleaning Nan and filling
clean=weather_data.dropna(subset=['weather_description'])
weather_data['weather_description'].fillna(weather_data['weather_description'].mean(),inplace=True)
print(weather_data.isna().sum())



# Adjust the features accordingly based on your dataset
features = weather_data[['temperature', 'humidity', 'pressure','wind_speed','wind_direction','weather_description']]
target = weather_data['weather_description']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
print(predictions)

# Evaluate the model
new_data = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {new_data}')

# Plot the actual vs. predicted values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Future Weather')
plt.ylabel('Predicted Future Weather')
plt.title('Actual vs. Predicted Future Weather')
plt.show()


if new_data >= 0:
    print("weather is clear")
elif new_data >= 1:
    print("weather is few clouds")
elif new_data >= 2:
    print("weather is scattered clouds")
elif new_data >= 3:
    print("weather is light rain")
elif new_data >= 4:
    print("weather is broken clouds")
elif new_data >= 5:
    print("weather is overcast clouds")
elif new_data >= 6:
    print("weather is proximity thunderstorm")
elif new_data >= 7:
    print("weather is mist")
elif new_data >= 8:
    print("weather is haze")
elif new_data >= 9:
    print("weather is heavy intensity rain")