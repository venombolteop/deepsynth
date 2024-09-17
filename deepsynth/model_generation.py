# Telegram :- @K_4ip
import tensorflow as tf

class ModelGeneration:
    def __init__(self, num_layers, units_per_layer):
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer

    def generate_model(self):
        model = tf.keras.Sequential()
        for _ in range(self.num_layers):
            model.add(tf.keras.layers.Dense(self.units_per_layer, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
