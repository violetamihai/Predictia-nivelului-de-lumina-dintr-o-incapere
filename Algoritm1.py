### Moving Average ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error


# Încarcă datele din fișierul CSV
data = pd.read_csv('data.csv', parse_dates=[0], infer_datetime_format=True, header=None,
                   names=['Timestamp', 'Type', 'Value'])

# Filtrare date pentru tipul 'light'
light_data = data[data['Type'] == 'light']


# Extrage valorile
light_values = light_data['Value'].values

# Aplică filtru median cu fereastră de 3 pentru preprocesare
preprocessed_light_values = medfilt(light_values, kernel_size=3)

#inversam vectorul, cronologic, e pe dos, prima intrare din fisier e ultima
preprocessed_light_values[:] = preprocessed_light_values[::-1]

# vector indici
X = np.arange(len(preprocessed_light_values)).reshape(-1, 1)

# moving average
window_size = 5
future_predictions = np.convolve(preprocessed_light_values, np.ones(window_size) / window_size, mode='valid')

# Extinde seria temporală pentru următoarele 100 de valori
for _ in range(100):
    # Prezice următoarea valoare folosind ultimele valori și media mobilă
    next_value = np.mean(preprocessed_light_values[-window_size:])

    # Adaugă valoarea prezisă la listă
    future_predictions = np.append(future_predictions, next_value)

    # Actualizează seria temporală de intrare pentru următoarea iterație
    preprocessed_light_values = np.append(preprocessed_light_values, next_value)

# extinde vectorul de indici
X_extended = np.arange(len(preprocessed_light_values)).reshape(-1, 1)


y_true = preprocessed_light_values[-len(future_predictions):]
mse = mean_squared_error(y_true, future_predictions)
print(f'Mean Squared Error: {mse}')

# grafice
plt.figure(figsize=(12, 6))
plt.plot(X_extended, preprocessed_light_values, label='Valori adevarate (Preprocesate)', color='blue')
plt.plot(X_extended[len(X_extended) - len(future_predictions):], future_predictions, color='orange',
         label=f'Valori prezise')


plt.xlabel('Timp', fontsize=12)
plt.ylabel('Valoare Luminozitate (Preprocesată)', fontsize=12)
plt.title('Predicție Valoare Luminozitate cu Moving Average', fontsize=16)
plt.legend()
plt.show()
