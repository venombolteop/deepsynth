# Telegram :- @K_4ip
import unittest
from deepsynth.nas import NeuralArchitectureSearch

class TestNeuralArchitectureSearch(unittest.TestCase):
    def test_model_creation(self):
        nas = NeuralArchitectureSearch(search_space=[{'layers': 2, 'units': 64}])
        model = nas.create_model(2, 64)
        self.assertEqual(len(model.layers), 3)  # 2 hidden layers + 1 output layer
