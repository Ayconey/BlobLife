import tensorflow as tf
import numpy as np

# Tworzenie modelu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Warstwa wejściowa o długości 16
    tf.keras.layers.Dense(32, activation='relu'),  # Pierwsza warstwa gęsta
    tf.keras.layers.Dense(16, activation='relu'),  # Druga warstwa gęsta
    tf.keras.layers.Dense(4, activation='softmax')  # Trzecia warstwa gęsta z 4 wyjściami
])

# Kompilacja modelu (możesz dostosować optymalizator i funkcję straty)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Przykładowe dane wejściowe o długości 16
sample_input = np.random.rand(1, 16)  # Dodajemy wymiar batcha (1) i długość wejścia (16)

# Przewidywanie akcji dla przykładowego wejścia
final_input = np.array([sample_input,sample_input])
predicted_actions = model.predict([[1]])

# Wyświetlenie przewidywanych akcji
print(predicted_actions)
print(max(predicted_actions))