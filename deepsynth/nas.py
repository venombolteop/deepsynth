# by venombolteop

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class NeuralArchitectureSearch:
    def __init__(self, search_space):
        self.search_space = search_space

    def create_model(self, layers, units):
        model = Sequential()
        for _ in range(layers):
            model.add(Dense(units=units, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def search_best_model(self, X_train, y_train):
        best_model = None
        best_score = 0
        for config in self.search_space:
            model = self.create_model(config['layers'], config['units'])
            model.fit(X_train, y_train, epochs=10, verbose=0)
            score = model.evaluate(X_train, y_train, verbose=0)[1]
            if score > best_score:
                best_score = score
                best_model = model
        return best_model
