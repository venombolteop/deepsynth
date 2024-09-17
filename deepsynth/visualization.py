# Telegram :- @VenomOwners
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    @staticmethod
    def plot_model_performance(history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_search_progress(search_results):
        sns.lineplot(data=search_results, x='iteration', y='performance')
        plt.title('Hyperparameter Search Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.show()
