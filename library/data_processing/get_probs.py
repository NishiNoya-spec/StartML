import numpy as np

class SigmoidProb:
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def sigmoid(self, M):
        return 1 / (1 + np.exp(-M))

    def normalize_probabilities(self, probabilities):
        # Нормализуем вероятности в диапазоне от 0 до 1
        min_prob = np.min(probabilities)
        max_prob = np.max(probabilities)
        normalized_probs = (probabilities - min_prob) / (max_prob - min_prob)
        return normalized_probs

    def check_probabilities(self):
        # Получаем отступы от гиперплоскости
        M = self.model.decision_function(self.X_test)
        # Прогоняем отступы через сигмоидную функцию активации
        probabilities = self.sigmoid(M)

        # Находим минимальное и максимальное значение среди всех вероятностей
        overall_min = np.min(probabilities)
        overall_max = np.max(probabilities)

        # Проверяем, находятся ли вероятности в диапазоне от 0 до 1
        if 0 <= overall_min <= 1 and 0 <= overall_max <= 1:
            print("Вероятности находятся в диапазоне от 0 до 1.")
            return probabilities
        else:
            print("Вероятности не находятся в диапазоне от 0 до 1. Производим нормализацию.")
            return self.normalize_probabilities(probabilities)


### *********************

#~ from library.data_processing.get_probs import SigmoidProb

#~ checker = SigmoidProb(pipe, X_test)
#~ probabilities = checker.check_probabilities()
#~ probabilities_norm = checker.normalize_probabilities(probabilities)

### *********************