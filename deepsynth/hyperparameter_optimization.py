# Telegram :- @K_4ip
from hyperopt import hp, tpe, fmin

class HyperparameterOptimization:
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def optimize(self, space):
        best = fmin(self.objective_function, space, algo=tpe.suggest, max_evals=100)
        return best
