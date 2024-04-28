import seaborn as sns

class OutlierHandler:
    def __init__(self, df):
        self.df = df
        self.outliers = []

    def find_outliers(self, feature, tentacle_length=1.5):
        """
        Находит выбросы в указанном признаке и сохраняет их в self.outliers.

        Параметры:
        - feature: str, имя признака для анализа выбросов.
        - tentacle_length: float, длина усиков в интервале IQR, по умолчанию 1.5.
        """
        q_low = self.df[feature].quantile(0.25)
        q_high = self.df[feature].quantile(0.75)
        iqr = q_high - q_low
        lower_tentacle = q_low - tentacle_length * iqr
        upper_tentacle = q_high + tentacle_length * iqr
        outliers = self.df[(self.df[feature] < lower_tentacle) | (self.df[feature] > upper_tentacle)][feature]
        self.outliers = outliers
        return outliers

    def visualize_outliers(self, feature, tentacle_length=1.5):
        """
        Визуализирует выбросы в указанном признаке с помощью boxplot.

        Параметры:
        - feature: str, имя признака для визуализации выбросов.
        - tentacle_length: float, длина усиков в интервале IQR, по умолчанию 1.5.
        """
        q_low = self.df[feature].quantile(0.25)
        q_high = self.df[feature].quantile(0.75)
        iqr = q_high - q_low
        lower_tentacle = q_low - tentacle_length * iqr
        upper_tentacle = q_high + tentacle_length * iqr
        sns.boxplot(x=self.df[feature], whis=tentacle_length)



"""
### Пример использования

import pandas as pd
import seaborn as sns

# Создаем экземпляр класса OutlierHandler
handler = OutlierHandler(df)

# Находим выбросы и визуализируем их
outliers = handler.find_outliers('d1')
handler.visualize_outliers('d1')

# Отбрасываем выбросы из DataFrame
cleaned_df = df[~df['d1'].isin(outliers)]

"""