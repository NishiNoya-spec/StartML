import torch


def function03(x: torch.Tensor, y: torch.Tensor):
    n_steps = 2000
    step_size = 1e-2

    # Определяем количество признаков (столбцов) в датасете x
    n_features = x.shape[1]

    # Создаем тензор с весами, инициализируем их из равномерного распределения
    weights = torch.rand(n_features, dtype=torch.float32, requires_grad=True)

    for i in range(n_steps):
        # Предсказание линейной регрессии
        y_pred = x @ weights

        # Вычисляем MSE
        mse = torch.mean((y_pred - y) ** 2)

        if i < 20 or i % 50 == 0:
            print(f'MSE на шаге {i + 1} {mse.item():.5f}')

        # Вычисляем градиенты
        mse.backward()

        # Обновляем веса с использованием градиентного спуска
        with torch.no_grad():
            weights -= step_size * weights.grad

        # Обнуляем градиенты
        weights.grad.zero_()

    return weights


if __name__ == '__main__':
    n_features = 2  # Размерность указана для примера, не нужно использовать её в коде
    n_objects = 300

    w_true = torch.randn(n_features)
    X = (torch.rand(n_objects, n_features) - 0.5) * 5
    Y = X @ w_true + torch.randn(n_objects) / 2

    result = function03(X, Y)
    print("Обученные веса:", result)


