def get_redundant_pairs(df):
    """
    Функция для определения пар признаков, для которых корреляция уже была рассчитана.

    Параметры:
    - df: DataFrame, содержащий признаки.

    Возвращает:
    - pairs_to_drop: множество, содержащее пары признаков, для которых корреляция уже была рассчитана.
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    """
    Функция для получения верхних абсолютных корреляций между признаками в DataFrame.

    Параметры:
    - df: DataFrame, содержащий признаки.
    - n: int, количество верхних корреляций для возврата (по умолчанию 5).

    Возвращает:
    - au_corr[:n]: Series, содержащий верхние абсолютные корреляции между признаками.
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:n]

def print_top_abs_correlations(df, n=5):
    """
    Функция для вывода верхних абсолютных корреляций между признаками в DataFrame.

    Параметры:
    - df: DataFrame, содержащий признаки.
    - n: int, количество верхних корреляций для вывода (по умолчанию 5).
    """
    print("Top Absolute Correlations:")
    top_corr = get_top_abs_correlations(df, n).reset_index()
    top_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
    print(top_corr)

# Пример использования:
# print_top_abs_correlations(df[numeric_columns], 10)

# ==============================================================>>

# Удаление признаков, в которых корреляция превышает threshold

def remove_highly_correlated_features(dataset, threshold):
    """
    Удаляет признаки с высокой корреляцией из датасета.

    Параметры:
    - dataset: DataFrame, исходный датасет.
    - threshold: float, порог корреляции, при котором признаки будут удалены.

    Возвращает:
    - dataset: DataFrame, датасет с удаленными признаками.
    """
    col_corr = set()  # Множество названий удаленных столбцов
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # Получение названия столбца
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # Удаление столбца из датасета
    return dataset

# Пример использования
# df = remove_highly_correlated_features(df, 0.9)


