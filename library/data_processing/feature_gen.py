import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def generate_polynomial_features(X, numeric_columns, degrees):
    """
    Генерирует полиномиальные признаки заданной степени для указанных числовых столбцов.

    Параметры:
    - X: DataFrame с исходными признаками.
    - numeric_columns: список названий числовых столбцов, для которых нужно создать полиномиальные признаки.
    - degrees: список степеней, для которых нужно создать полиномиальные признаки.

    Возвращает:
    - X_poly: DataFrame с добавленными полиномиальными признаками.
    """
    X_poly = X.copy()
    for col in numeric_columns:
        if col != 'Surge_Pricing_Type':
            for degree in degrees:
                polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
                poly_data = polynomial_features.fit_transform(X[[col]])
                poly_columns = [f"{col}_{degree}^{i}" for i in range(1, degree+1)]
                poly_df = pd.DataFrame(poly_data[:,1:], columns=poly_columns, index=X.index)
                X_poly = pd.concat([X_poly, poly_df], axis=1)
    return X_poly

#! ==============================================================>>

from itertools import combinations_with_replacement

def generate_interactions_features(X, numeric_columns, degrees):
    """
    Генерирует новые признаки, учитывая пересечения между признаками.

    Параметры:
    - X: DataFrame с исходными признаками.
    - numeric_columns: список названий числовых столбцов, для которых нужно создать новые признаки.
    - degrees: список степеней, для которых нужно создать новые признаки.

    Возвращает:
    - X_interactions: DataFrame с добавленными признаками, учитывающими пересечения между исходными признаками.
    """
    X_interactions = X.copy()
    for degree in degrees:
        for combo in combinations_with_replacement(numeric_columns, 2):
            for col1, col2 in [combo, combo[::-1]]:
                new_col_name = f"{col1}_x_{col2}_{degree}"
                interaction = X[col1] ** degree * X[col2] ** degree
                X_interactions[new_col_name] = interaction
    return X_interactions

