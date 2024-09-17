# DeepSynth Documentation

**DeepSynth** is a Python package designed to automate the synthesis of deep learning models using advanced techniques like Neural Architecture Search (NAS), Hyperparameter Optimization, Transfer Learning, and more. It simplifies the process of building high-performance models with minimal manual intervention.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Transfer Learning](#transfer-learning)
  - [Model Generation](#model-generation)
  - [Visualization](#visualization)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install DeepSynth and its dependencies, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install DeepSynth directly using pip:

```bash
pip install deepsynth
```

## Features

- **Neural Architecture Search (NAS)**: Automatically explores various model architectures to find the optimal one for your task.
- **Hyperparameter Optimization**: Fine-tunes model hyperparameters using techniques like Bayesian optimization or genetic algorithms.
- **Transfer Learning**: Utilizes pre-trained models and adapts them for specific tasks with minimal adjustments.
- **Model Generation**: Generates new model architectures based on user-defined constraints and datasets.
- **Visualization Tools**: Provides tools for visualizing model architectures, performance metrics, and optimization progress.

## Usage

### Neural Architecture Search (NAS)

```python
from deepsynth.nas import NeuralArchitectureSearch
import numpy as np

# Sample data preparation
X_train = np.random.rand(1000, 20)  # 1000 samples, 20 features
y_train = np.random.randint(2, size=1000)  # Binary target

search_space = [
    {'layers': 2, 'units': 64},
    {'layers': 3, 'units': 128},
    {'layers': 4, 'units': 256}
]
nas = NeuralArchitectureSearch(search_space)
best_model = nas.search_best_model(X_train, y_train)
print("Best model found by NAS:")
best_model.summary()
```

### Hyperparameter Optimization

```python
from deepsynth.hyperparameter_optimization import HyperparameterOptimization
from hyperopt import hp

def objective_function(params):
    from deepsynth.model_generation import ModelGeneration
    model = ModelGeneration(params['num_layers'], params['units_per_layer']).generate_model()
    model.fit(X_train, y_train, epochs=5, verbose=0)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    return {'loss': loss, 'status': 'ok'}

space = {
    'num_layers': hp.choice('num_layers', [2, 3, 4]),
    'units_per_layer': hp.choice('units_per_layer', [32, 64, 128])
}
optimizer = HyperparameterOptimization(objective_function)
best_params = optimizer.optimize(space)
print("Best hyperparameters found:")
print(best_params)
```

### Transfer Learning

```python
from deepsynth.transfer_learning import TransferLearning

transfer_learning = TransferLearning()
model = transfer_learning.load_model()
print("Transfer learning model:")
model.summary()
```

### Model Generation

```python
from deepsynth.model_generation import ModelGeneration

num_layers = 3
units_per_layer = 128
model_generator = ModelGeneration(num_layers, units_per_layer)
model = model_generator.generate_model()
print("Generated model:")
model.summary()
```

### Visualization

```python
from deepsynth.visualization import Visualization

# Dummy training history for demonstration
history = {
    'accuracy': [0.1, 0.3, 0.5, 0.7, 0.9],
    'val_accuracy': [0.15, 0.35, 0.55, 0.75, 0.85]
}
Visualization.plot_model_performance(history)
```

## Examples

For additional examples and use cases, refer to the `examples/` directory in the package. The `example_usage.py` file demonstrates various functionalities of DeepSynth.

## Contributing

Contributions to DeepSynth are welcome! Please follow these steps to contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your changes.
3. Commit your changes and push them to your fork.
4. Open a pull request describing your changes.

## License

DeepSynth is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

For further questions or support, please open an issue on the [GitHub repository](https://github.com/venombolteop/deepsynth) or contact the maintainers.
