# Telegram :- @VenomOwners
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class TransferLearning:
    def __init__(self, base_model='VGG16'):
        self.base_model = base_model

    def load_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
