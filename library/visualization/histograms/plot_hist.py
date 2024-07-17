import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(df, columns, titles=None, rotation=0, fig_size=(16, 10)):
    """
    Функция для визуализации гистограмм для указанных столбцов DataFrame.

    Параметры:
    - df: DataFrame, содержащий данные.
    - columns: список столбцов, для которых нужно построить гистограммы.
    - titles: список заголовков для каждого подграфика. По умолчанию None.
    - rotation: угол поворота подписей категорий на гистограммах. По умолчанию 0.
    - fig_size: кортеж (width, height) для размера фигуры. По умолчанию (16, 10).
    """
    num_plots = len(columns)
    num_rows = int(np.ceil(num_plots / 2))

    fig, axes = plt.subplots(num_rows, 2, figsize=fig_size)

    # Преобразование axes в одномерный массив, если num_plots = 1
    if num_plots == 1:
        axes = np.array([axes])

    for i, col in enumerate(columns):
        row_idx = i // 2
        col_idx = i % 2

        ax = axes[row_idx, col_idx]

        values = df[col].value_counts()
        my_cmap = plt.get_cmap('viridis')
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        ax.bar(values.index, values.values, color=my_cmap(rescale(values.values)))
        ax.set_title(titles[i] if titles else col)
        ax.set_xticklabels(values.index, rotation=rotation)

    # Удаление пустых подграфиков, если num_plots нечетное
    if num_plots % 2 != 0:
        fig.delaxes(axes[num_rows - 1, 1])

    fig.tight_layout()
    plt.show()


###^ Пример построения гистограмм для 4-ех признаков 

### Распределение магазинов по городам/штатам/типам/кластерам

fig = plt.figure()
fig.set_size_inches(16, 10)

# По городам 

ax_1 = fig.add_subplot(2, 2, 1)

my_cmap = plt.get_cmap('viridis')
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

for_hist = stores.groupby('city').size()
plt.bar(for_hist.index, for_hist.values, color=my_cmap(rescale(for_hist.values)))
plt.xticks(rotation=45, size=10)

### по штатам
ax_2 = fig.add_subplot(2, 2, 2)

for_hist = stores.groupby('state').size()
plt.bar(for_hist.index, for_hist.values, color=my_cmap(rescale(for_hist.values)))
plt.xticks(rotation=30, size=10)

### по типу
ax_3 = fig.add_subplot(2, 2, 3)

for_hist = stores.groupby('type').size()
plt.bar(for_hist.index, for_hist.values, color=my_cmap(rescale(for_hist.values)))
plt.xticks(rotation=30, size=10)

### по кластеру
ax_4 = fig.add_subplot(2, 2, 4)

for_hist = stores.groupby('cluster').size()
plt.bar(for_hist.index, for_hist.values, color=my_cmap(rescale(for_hist.values)))
plt.xticks(rotation=30, size=10)

fig.tight_layout()

ax_1.set(title = 'Распределение магазинов по городам')
ax_2.set(title = 'Распределение магазинов по штатам')
ax_3.set(title = 'Распределение магазинов по типам')
ax_4.set(title = 'Распределение магазинов по кластерам')

plt.show()

