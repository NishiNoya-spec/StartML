import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
import pandas as pd
from IPython.display import display

def visualize_decision_regions(clfs_dict, X, Y):
    """
    Визуализирует решающие поверхности для каждой модели классификации из словаря обученных моделей,
    а также выводит сводную таблицу с метриками классификации для каждой модели.

    Параметры:
    - clfs_dict: dict, словарь обученных моделей классификации, где ключ - имя модели, а значение - сама модель.
    - X: массив признаков, shape (n_samples, n_features), признаки для предсказания.
    - Y: массив меток, shape (n_samples,), истинные метки классов.

    Возвращает:
    - None: функция выводит графики решающих поверхностей и сводную таблицу с метриками классификации.
    """
    num_models = len(clfs_dict)
    num_rows = num_models // 3 + (num_models % 3 > 0)
    gs = gridspec.GridSpec(num_rows, 3)
    fig = plt.figure(figsize=(15, 5 * num_rows))

    labels = [clf_name for clf_name in clfs_dict.keys()]
    all_clfs = [clf for clf in clfs_dict.values()]
    metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    for clf, lab, grd in zip(all_clfs, labels, itertools.product(range(num_rows), range(3))):
        ax = plt.subplot(gs[grd[0], grd[1]])
        plot_decision_regions(X=X, y=Y, clf=clf, legend=2, ax=ax)
        plt.title(lab)
        
        # Вычисляем метрики классификации
        y_pred = clf.predict(X)
        report = classification_report(Y, y_pred, output_dict=True)
        
        # Добавляем метрики в словарь
        metrics['Model'].append(lab)
        metrics['Accuracy'].append(report['accuracy'])
        metrics['Precision'].append(report['macro avg']['precision'])
        metrics['Recall'].append(report['macro avg']['recall'])
        metrics['F1-Score'].append(report['macro avg']['f1-score'])

    plt.tight_layout()
    plt.show()
    
    # Создаем DataFrame с метриками и сортируем по убыванию Accuracy
    metrics_df = pd.DataFrame(metrics).sort_values(by='Accuracy', ascending=False)
    
    # Выводим сводную таблицу
    display("Classification Report:")
    display(metrics_df)

