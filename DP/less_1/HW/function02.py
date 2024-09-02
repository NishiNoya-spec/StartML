import numpy as np

import torch
from torch import nn

def function02(dataset: torch.Tensor):
    # Определяем размерность тензора с весами, она должна совпадать с количеством признаков (столбцов) в датасете
    num_features = dataset.shape[1]

    # Создаем тензор с весами из равномерного распределения на отрезке [0, 1)
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)

    return weights

if __name__ == '__main__':

    example_dataset = torch.rand(3, 5)

    weights = function02(example_dataset)
    print("Сгенерированные веса:", weights)