### 7. Определение коррелириющих признаков. Корреляция Пирса 
 
def get_redundant_pairs(df): 
    """ 
    Функция для определения пар признаков, для которых корреляция уже была рассчитана. 
 
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
 
    """ 
    au_corr = df.corr().abs().unstack() 
    labels_to_drop = get_redundant_pairs(df) 
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False) 
    return au_corr[:n] 
 

def print_top_abs_correlations(df, n=5): 
    """ 
    Функция для вывода верхних абсолютных корреляций между признаками в DataFrame. 
 
    """ 
    print("Top Absolute Correlations:") 
    top_corr = get_top_abs_correlations(df, n).reset_index() 
    top_corr.columns = ['Feature 1', 'Feature 2', 'Correlation'] 
    print(top_corr) 
 

def remove_highly_correlated_features(df, target="class", threshold=0.7): 
 
    ### Удаляет признаки с высокой корреляцией из датасета. (При отбрросе фичи из двух коррелирующих A и B - будет отброшена B) 
 
    df = df.copy() 
 
    X = df.drop(target, axis=1) 
    y = df[target] 
 
    numeric_columns, categorical_columns = numeric_categorical_columns(X)  
 
    dataset = X[numeric_columns] 
    initial_columns_count = dataset.shape[1] 
    col_corr = set()  # Множество названий удаленных столбцов 
    corr_matrix = dataset.corr() 
    for i in range(len(corr_matrix.columns)): 
        for j in range(i): 
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr): 
                colname = corr_matrix.columns[i]  # Получение названия столбца 
                col_corr.add(colname) 
                if colname in dataset.columns: 
                    del dataset[colname]  # Удаление столбца из датасета 
 
    removed_high_corr_count = initial_columns_count - dataset.shape[1] 
    print(f"Кол-во удаленных коррелириующих столбцов: {removed_high_corr_count}") 
 
    dataset = pd.concat([dataset, X[categorical_columns], y], axis=1) 
 
    return dataset 
 
 
import dask.dataframe as dd 
from dask.distributed import Client 
from multiprocessing import Pool, cpu_count 
import logging 
from sklearn.feature_selection import f_classif
 
def remove_highly_correlated_features_with_anova(df, num_threads=None, target="class", threshold=0.7): 
    """ 
    Удаляет признаки с высокой корреляцией из датасета, оставляя более значимые для целевой переменной признаки. 
    (Значимость фичи определяется с помощью алгоритма Anova-фильтрации (F-value - Коэффициент означает способность признака разделять уровни целевой переменной)). 
    """ 
    df = df.copy() 
     
    X = df.drop(target, axis=1) 
    y = df[target] 
     
    numeric_columns, categorical_columns = numeric_categorical_columns(X) 
    dataset = X[numeric_columns] 
    initial_columns_count = dataset.shape[1] 
     
    col_corr = set()  # Множество названий удаленных столбцов 
     
    # Настройка логгера 
    logging.basicConfig(level=logging.INFO) 
    logger = logging.getLogger(__name__) 
     
    # Определение количества потоков 
    if num_threads is None: 
        corr_matrix = dataset.corr().abs() 
    elif num_threads == "all_available": 
        client = Client() 
        num_threads = client.nthreads() 
        num_threads = len(num_threads) 
        logger.info(f"Используется количество потоков: {num_threads}") 
    else: 
        client = Client(n_workers=num_threads) 
        logger.info(f"Используется количество потоков: {num_threads}") 
 
    if num_threads is not None: 
        # Используем dask для вычисления корреляционной матрицы 
        ddf = dd.from_pandas(dataset, npartitions=num_threads) 
        # Расчет корреляционной матрицы 
        corr_matrix = ddf.corr().compute().abs() 
     
    # Вычисляем значимость признаков с помощью ANOVA 
    anova_scores, p_values = f_classif(dataset, y)
    anova_scores = pd.Series(anova_scores, index=dataset.columns) 
     
    # Создаем список пар коррелирующих признаков 
    correlated_pairs = [] 
    for i in tqdm(range(len(corr_matrix.columns)), desc="Mapping"): 
        for j in range(i): 
            if corr_matrix.iloc[i, j] >= threshold: 
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j])) 
     
    # Обрабатываем пары коррелирующих признаков 
    for colname_i, colname_j in tqdm(correlated_pairs, desc='Processing correlated pairs'): 
        if colname_i in col_corr or colname_j in col_corr: 
            continue 
        # Сравниваем значимость признаков по F-value 
        if anova_scores[colname_i] < anova_scores[colname_j]: 
            col_corr.add(colname_i) 
        else: 
            col_corr.add(colname_j) 
     
    dataset.drop(columns=list(col_corr), inplace=True) 
     
    removed_high_corr_count = initial_columns_count - dataset.shape[1] 
    print(f"Кол-во удаленных коррелирующих столбцов: {removed_high_corr_count}") 
     
    dataset = pd.concat([dataset, X[categorical_columns], y], axis=1) 
     
    return dataset


