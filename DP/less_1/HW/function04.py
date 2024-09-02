import torch
from torch import nn

def function04(x: torch.Tensor, y: torch.Tensor):
    n_steps = 2000
    step_size = 1e-2

    # Определяем количество признаков (столбцов) в датасете x
    n_features = x.shape[1]

    # Создаем полносвязный слой
    layer = nn.Linear(in_features=n_features, out_features=1)

    for i in range(n_steps):
        # Предсказание линейной регрессии
        y_pred = layer(x).view(-1)  # Преобразуем к одномерному тензору

        # Вычисляем MSE
        mse = torch.mean((y_pred - y) ** 2)

        if i < 20 or i % 50 == 0:
            print(f'MSE на шаге {i + 1} {mse.item():.5f}')

        if mse < 0.3:
            print('break', mse)
            break

        # Вычисляем градиенты
        mse.backward()

        # Обновляем веса с использованием градиентного спуска
        with torch.no_grad():
            layer.weight -= step_size * layer.weight.grad
            layer.bias -= step_size * layer.bias.grad

        # Обнуляем градиенты
        layer.zero_grad()

    return layer

if __name__ == '__main__':
    n_features = 2
    n_objects = 300

    w_true = torch.randn(n_features)
    X = (torch.rand(n_objects, n_features) - 0.5) * 5
    Y = X @ w_true + torch.randn(n_objects) / 2

    trained_layer = function04(X, Y)
    print("Обученные веса:", trained_layer.weight)
    print("Обученное смещение (bias):", trained_layer.bias)