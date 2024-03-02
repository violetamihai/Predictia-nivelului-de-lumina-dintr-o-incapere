import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from statsmodels.tsa.seasonal import STL
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error

# datele din fisier se incarca intr-o variabila data
data = pd.read_csv('data.csv', parse_dates=[0], infer_datetime_format=True, header=None, names=['Timestamp', 'Type', 'Value'])

# pastram valorile pt lumina
light_data = data[data['Type'] == 'light']
light_values = light_data['Value'].values

# aplicam filtru median cu fereastra de 3
preprocessed_light_values = medfilt(light_values, kernel_size=3)

#inversam vectorul, cronologic, e pe dos, prima intrare din fisier e ultima
preprocessed_light_values[:] = preprocessed_light_values[::-1]

# aplicam STL decomposition si salvam cele 3 componente in variabile
seasonal_period = 240  #perioada este de 24h si facem 10 masuratori pe ora
stl = STL(preprocessed_light_values, period=seasonal_period)
result = stl.fit()
trend, seasonal, residual = result.trend, result.seasonal, result.resid

# prezicem urmatoarele 600 de valori
window_size =5
future_trend = np.convolve(trend[-window_size:], np.ones(window_size) / window_size, mode='valid')
future_seasonal = np.convolve(seasonal[-window_size:], np.ones(window_size) / window_size, mode='valid')

future_trend_component = trend[-600:]
future_seasonal_component = seasonal[-600:]

#combinam trend cu seasonal
future_predictions = future_trend_component + future_seasonal_component

#plotare date preprocesate si inainte
X = np.arange(len(preprocessed_light_values)).reshape(-1, 1)
plt.figure(figsize=(12, 6))
plt.plot(X, preprocessed_light_values, label='dupa filtru median', color='blue')
plt.plot(X, light_values[::-1], label='inainte filtru median', color='orange',linestyle='--')
plt.xlabel('Indice masuratoare', fontsize=12)
plt.ylabel('Valoare luminozitate', fontsize=12)
plt.title('Aplicare filtru median', fontsize=16)
plt.legend()
plt.show()

# grafice predictie
plt.figure(figsize=(12, 6))
plt.plot(preprocessed_light_values, label='Valori adevarate preprocesate', color='blue')
plt.plot(np.arange(len(preprocessed_light_values), len(preprocessed_light_values) + 600), future_predictions, color='orange', label='Predictie STL')
plt.xlabel('Indice masuratoare', fontsize=12)
plt.ylabel('Valoare luminozitate', fontsize=12)
plt.title('Predictie STL', fontsize=16)
plt.legend()
plt.show()

#grafic pentru afisarea componentelor
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(30, 16), sharex=True)

# valorile adevarate
ax1.plot(preprocessed_light_values, label='Valori adevarate preprocesate', color='blue')
ax1.set_ylabel('Valori adevarate preprocesate', fontsize=12)
ax1.legend()

#trend
ax2.plot(np.arange(len(preprocessed_light_values)), trend, label='Componenta trend', color='green', linestyle='--')
ax2.set_ylabel('Componenta trend', fontsize=12)
ax2.legend()

#seasonal component
ax3.plot(np.arange(len(preprocessed_light_values)), seasonal, label='Componenta sezoniera', color='red', linestyle='--')
ax3.set_ylabel('Componenta sezoniera', fontsize=12)
ax3.legend()

#componenta reziduala
ax4.plot(np.arange(len(preprocessed_light_values)), residual, label='Componenta reziduala', color='purple', linestyle='--')
ax4.set_xlabel('Indice masuratoare', fontsize=12)
ax4.set_ylabel('Componenta reziduala', fontsize=12)
ax4.legend()

plt.suptitle('Descompunere STL in componente ', fontsize=16)
plt.show()

#citim din fisierul cu ultimele 2 zile date
file_path = 'last'
light_values = []

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if row[1] == 'light':
            light_values.append(int(row[2]))

#afisam realitatea ultimelor 2 zile si predictia
plt.figure(figsize=(12, 6))
#am inversat vectorul pentru ca el e citit pe invers din fisier
plt.plot(light_values[::-1], label='Real', color='blue')
plt.plot(np.arange(600), future_predictions, color='orange', label='Predictie STL')
plt.xlabel('Index ', fontsize=12)
plt.ylabel('Luminozitate', fontsize=12)
plt.title('Realitate vs Predictie', fontsize=16)
plt.legend()
plt.show()

#afisam MSE
real_values = light_values[::-1]
mse = mean_squared_error(real_values, future_predictions[:483])
print(f'Mean Squared Error (MSE): {mse}')