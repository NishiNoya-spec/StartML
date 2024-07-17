"""
Для отбрасывания константных и квазиконстантных признаков используется метод VarianceThreshold
из библиотеки scikit-learn. Этот метод удаляет признаки, чья дисперсия меньше заданного порога.
Формула для расчета дисперсии проста:

Var(X) = Σ (xᵢ - μ)² / n

Где:

Var(X) - дисперсия признака X.
Σ - сумма по всем элементам.
xᵢ - каждое отдельное значение признака.
μ - среднее значение признака.
n - количество значений признака.

"""

### 3. Отбрасываем константные и квазиконстантные признаки  
 
from sklearn.feature_selection import VarianceThreshold 
 
def drop_constant_features(df, target="class", threshold=0.05): 
    """ 
    Отбрасывает константные и квазиконстантные признаки из датасета. 
 
    """ 
    df = df.copy() 
 
    X = df.drop(target, axis=1) 
    y = df[target] 
 
    numeric_columns, categorical_columns = numeric_categorical_columns(X)  
 
    dataset = X[numeric_columns] 
    cutter = VarianceThreshold(threshold=threshold) 
    cutter.fit(dataset) 
    selected_features = cutter.get_support() 
    constant_cols = dataset.columns[~selected_features].tolist() 
    dataset = dataset.drop(columns=constant_cols, axis=1) 
    display(f"Кол-во отброшенных фичей по константному признаку: {len(constant_cols)}") 
    dataset = pd.concat([dataset, X[categorical_columns], y], axis=1) 
    return dataset
