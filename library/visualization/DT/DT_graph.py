import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot_decision_tree(decision_tree, feature_names, class_names):
    """
    Визуализирует дерево решений в виде графа.

    Параметры:
        decision_tree : обученная модель дерева решений
        feature_names : list
            Список названий признаков.
        class_names : list
            Список названий классов.
    """
    fig = plt.figure(figsize=(16, 10))

    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True
    )

    plt.show()