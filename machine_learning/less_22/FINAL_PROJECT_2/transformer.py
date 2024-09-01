import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, numeric_cols, target_name, threshold=5, noise_k=0.006):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_name = target_name
        self.threshold = threshold
        self.noise_k = noise_k
        self.ohe_encoders = {}
        self.target_encoders = {}
        self.mean_values = {}
        self.user_stats = {}  # Словарь для хранения статистики по пользователям
        self.dict_of_means = {}  # Словарь для хранения закодированных значений

    def fit(self, X, y):
        X_fit = X.copy()
        y_fit = y.copy()
        X_with_target = pd.concat((X_fit, y_fit), axis=1)

        # Распределение категориальных фичей по энкодерам
        for col in self.categorical_cols:
            if X_with_target[col].nunique() > self.threshold:
                # Target Encoding
                mean_target_values = X_with_target.groupby(col)[self.target_name].mean()
                self.dict_of_means[col] = mean_target_values + self.noise_k * np.random.randn(len(mean_target_values))
            else:
                encoder = OneHotEncoder(cols=[col], use_cat_names=True, handle_unknown='ignore')
                encoder.fit(X_with_target[[col]])
                self.ohe_encoders[col] = encoder

        # Вычисление статистики по пользователям
        user_count_views = X_with_target.groupby('user_id').size()
        user_means = X_with_target.groupby('user_id')['like_target'].mean()

        # Сохранение статистики в словари
        self.user_stats['views'] = user_count_views.to_dict()
        self.user_stats['means'] = user_means.to_dict()

        self.columns_ = X.columns  # Сохранение колонок для проверки в transform
        return self

    def transform(self, X):
        X_ = X.copy()

        # Проверка наличия колонок и добавление отсутствующих
        missing_cols = set(self.columns_) - set(X_.columns)
        for col in missing_cols:
            X_[col] = 0

        extra_cols = set(X_.columns) - set(self.columns_)
        if extra_cols:
            print(f"Warning: New columns in transform data: {extra_cols}")

        X_ = X_[self.columns_]
        df_orig = X_[self.numeric_cols]
        df_categorical = X_[self.categorical_cols]

        for col in self.categorical_cols:
            if col in self.dict_of_means:
                # Target Encoding
                df_categorical[col] = df_categorical[col].map(self.dict_of_means[col])
                df_categorical[col] = df_categorical[col].fillna(self.dict_of_means[col].mean())
            elif col in self.ohe_encoders:
                transformed_ohe = self.ohe_encoders[col].transform(df_categorical[[col]])
                df_categorical = df_categorical.join(transformed_ohe.iloc[:, 1:])  # Дроп первого столбца
                df_categorical.drop(columns=[col], inplace=True)

        X_ = pd.concat((df_orig, df_categorical), axis=1)

        # Добавление новых признаков
        X_ = self._create_additional_features(X_)
        return X_

    def _create_additional_features(self, X):
        # Обработка статистики по пользователям
        X['userViews'] = X['user_id'].map(self.user_stats['views'])
        X['userMeans'] = X['user_id'].map(self.user_stats['means']) + np.random.normal(0, self.noise_k, X.shape[0])

        # Заполнение пропусков для пользователей, которых не было в обучении
        overall_views_mean = np.mean(list(self.user_stats['views'].values()))
        overall_means_mean = np.mean(list(self.user_stats['means'].values()))
        X['userViews'] = X['userViews'].fillna(overall_views_mean)
        X['userMeans'] = X['userMeans'].fillna(overall_means_mean)

        return X


"""

# Пример использования
from sklearn.tree import DecisionTreeRegressor


def build_pipeline(categorical_cols, numeric_cols, target_name):
    custom_transformer = EncoderTransformer(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        target_name=target_name,
        threshold=5
    )

    col_transform = ColumnTransformer(
        transformers=[
            ('custom', custom_transformer, categorical_cols + numeric_cols)
        ],
        remainder='passthrough'
    )

    pipe = Pipeline([
        ('column_transformer', col_transform),
        ('scaler', StandardScaler()),  # Пример скалера, если нужно
        ('model', DecisionTreeRegressor(random_state=42))  # Замените модель на вашу
    ])

    return pipe


# Пример данных
X_train = pd.DataFrame({
    'user_id': [1, 2, 1, 2],
    'category1': ['a', 'b', 'a', 'c'],
    'numeric1': [1.0, 2.0, 3.0, 4.0]
})
y_train = pd.Series([0, 1, 0, 1])

# Инициализация и тренировка пайплайна
pipeline = build_pipeline(
    categorical_cols=['category1'],
    numeric_cols=['numeric1'],
    target_name='target'
)
pipeline.fit(X_train, y_train)


"""