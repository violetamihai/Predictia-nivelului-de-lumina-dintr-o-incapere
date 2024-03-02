import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.signal import medfilt

# Load data from the CSV file
data = pd.read_csv('data.csv', parse_dates=[0], infer_datetime_format=True, header=None, names=['Timestamp', 'Type', 'Value'])

# Filter data for 'light' type
light_data = data[data['Type'] == 'light']

# Sort data based on timestamp (if not already sorted)
light_data = light_data.sort_values(by='Timestamp')

# Extract the values and dates
light_values = light_data['Value'].values

# Apply median filter with window size 3 for preprocessing
preprocessed_light_values = medfilt(light_values, kernel_size=3)

# Create a feature matrix with consecutive numbers as features
X = np.arange(len(preprocessed_light_values)).reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, preprocessed_light_values, test_size=0.2, random_state=42)

# Create and train a Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = model.predict(X_test)

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Create a feature matrix for the next 10 values
future_X = np.arange(len(preprocessed_light_values), len(preprocessed_light_values) + 10).reshape(-1, 1)

# Generate predictions for the next 10 values
future_pred = model.predict(future_X)


plt.figure(figsize=(12, 6))
plt.plot(X, preprocessed_light_values, label='dupa filtru', color='blue')
plt.plot(X, light_values, label='inainte de filtru', color='orange',linestyle='--')
plt.xlabel('Indice masuratoare', fontsize=12)
plt.ylabel('Luminozitate', fontsize=12)
plt.title('Preprocesare cu filtru median', fontsize=16)
plt.legend()
plt.show()
# Plot the actual values
plt.figure(figsize=(12, 6))
plt.plot(X, preprocessed_light_values, label='Actual Values (Preprocessed)', color='blue')

# Plot the training set points and predicted values


# Plot the future predictions
plt.plot(future_X, future_pred, label='Indice masuratoare', color='orange', linestyle='--')

# Set labels and title
plt.xlabel('Indice masuratoare', fontsize=12)
plt.ylabel('Luminozitate', fontsize=12)
plt.title('Gradient Boosting Regressor ', fontsize=16)
plt.legend()
plt.show()
