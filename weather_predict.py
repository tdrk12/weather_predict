import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

data = {
    2014: [0, 2, 5, 10, 15, 20, 25, 22, 18, 12, 5, 2],
    2015: [1, 3, 6, 11, 16, 21, 26, 23, 19, 13, 6, 3],
    2016: [0, 2, 5, 10, 15, 20, 25, 22, 18, 12, 5, 2],
    2017: [2, 4, 7, 12, 17, 22, 27, 24, 20, 14, 7, 4],
    2018: [1, 3, 6, 11, 16, 21, 26, 23, 19, 13, 6, 3],
    2019: [0, 2, 5, 10, 15, 20, 25, 22, 18, 12, 5, 2],
    2020: [2, 4, 7, 12, 17, 22, 27, 24, 20, 14, 7, 4],
    2021: [1, 3, 6, 11, 16, 21, 26, 23, 19, 13, 6, 3],
    2022: [0, 2, 5, 10, 15, 20, 25, 22, 18, 12, 5, 2],
    2023: [2, 4, 7, 12, 17, 22, 27, 24, 20, 14, 7, 4],
    2024: [1, 3, 6, 11, 16, 21, 26, 23, 19, 13, 6, 3]
}

x_train = []
y_train = []
for year in range(2014, 2024):
    x_train.append(data[year])
    y_train.append(data[year + 1])

x_train = np.array(x_train)
y_train = np.array(y_train)

model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(12,)),
    keras.layers.Dense(12)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

last_year = np.array([data[2024]])
prediction = model.predict(last_year)[0]

with open('prediction.txt', 'w') as f:
    for temp in prediction:
        f.write(str(temp) + '\n')

def load_data(filename):
    with open(filename, 'r') as f:
        return [float(line) for line in f.readlines()]

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(months, prediction)
plt.show()

plt.plot(history.history['loss'])
plt.show()

print('loss:', np.mean(history.history['loss']))

model.save('weather_model.h5')

if os.path.exists('weather_model.h5'):
    choice = input('load model? (y/n): ')
    if choice == 'y':
        model = keras.models.load_model('weather_model.h5')