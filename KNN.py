import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Load data from the CSV file
data = pd.read_csv('data3', header=None, names=['Class', 'Value'])

# Separate class and value columns
class_data = data['Class'].str.extract('(\d+)', expand=False)
data['Class'] = class_data.astype(float)

# Handling missing values in the 'Value' column
imputer = SimpleImputer(strategy='mean')
data['Value'] = imputer.fit_transform(data['Value'].values.reshape(-1, 1))

# Split the data into training and testing sets
X = data[['Value']]
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a KNeighborsClassifier model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Create and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

