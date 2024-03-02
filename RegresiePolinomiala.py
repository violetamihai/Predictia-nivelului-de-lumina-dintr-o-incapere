import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
from scipy.signal import medfilt

# Load data from the CSV file
data_new = pd.read_csv('data2', parse_dates=[0], infer_datetime_format=True, header=None, names=['Timestamp', 'Type', 'Value'])

# Filter data for 'light' type
light_data_new = data_new[data_new['Type'] == 'light']

# Extract the last values for each timestamp
last_light_values_new = light_data_new.groupby('Timestamp')['Value'].last().values

# Apply median filter with window size 3 for preprocessing
preprocessed_light_values_new = medfilt(last_light_values_new, kernel_size=3)

# Create a feature matrix with consecutive numbers as features
X_new = np.arange(len(preprocessed_light_values_new))

# Define the sine function to fit
def sine_function(x, A, omega, phi, offset):
    return A * np.sin(omega * x + phi) + offset

# Fit the sine function to the data
params, _ = curve_fit(sine_function, X_new, preprocessed_light_values_new, p0=[1, 0.1, 0, np.mean(preprocessed_light_values_new)])

# Generate predictions using the fitted parameters
y_pred_all_new = sine_function(X_new, *params)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, preprocessed_light_values_new, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred_test = sine_function(X_test, *params)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred_test)
print(f'R-squared on Test Set: {r_squared}')

# Calculate Mean Squared Error (MSE)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Plot the actual values and sinusoidal regression predictions for the entire dataset
plt.figure(figsize=(12, 6))

# Plot training values in blue with dots 5 times smaller
plt.scatter(X_train, y_train, color='blue', label='Training Values', s=5)

# Plot test values in red with dots 5 times smaller
plt.scatter(X_test, y_test, color='red', label='Test Values', s=5)

# Plot the sinusoidal regression predictions for the entire dataset
plt.plot(X_new, y_pred_all_new, color='green', label='Sinusoidal Regression')

plt.xlabel('Index', fontsize=12)
plt.ylabel('Light Intensity', fontsize=12)
plt.title('Actual Values and Sinusoidal Regression Predictions', fontsize=16)

plt.legend()
plt.show()
