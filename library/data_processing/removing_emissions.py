"""
Выбор метода:
Изоляционный лес (Isolation Forest):

- Подходит для больших и сложных наборов данных.
- Хорош для данных с большим количеством признаков, где выбросы могут быть нелинейными.
- Когда важна автоматизация процесса.

Метод на основе квантилей (IQR):

- Подходит для небольших и средних наборов данных.
- Хорош для данных с небольшим количеством признаков.
- Когда важна интерпретируемость и контроль над процессом.

Рекомендации:
Если у вас большой и сложный набор данных с нелинейными выбросами, используйте изоляционный лес.
Если у вас относительно небольшой набор данных, где выбросы можно легко определить по отдельным
признакам, используйте метод на основе квантилей (IQR).

"""


import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#^ Метод на основе квантилей (IQR):

class OutlierHandler:
    def init(self, df):
        self.df = df   
        self.outliers_indices = set()

    def find_outliers(self, feature, tentacle_length=1.5):
        """
        Находит выбросы в указанном признаке и сохраняет их индексы в self.outliers_indices.

        Параметры:
        - feature: str, имя признака для анализа выбросов.
        - tentacle_length: float, длина усиков в интервале IQR, по умолчанию 1.5.
        """
        q_low = self.df[feature].quantile(0.25)
        q_high = self.df[feature].quantile(0.75)
        iqr = q_high - q_low
        lower_tentacle = q_low - tentacle_length * iqr
        upper_tentacle = q_high + tentacle_length * iqr
        outliers = self.df[(self.df[feature] < lower_tentacle) | (self.df[feature] > upper_tentacle)]
        self.outliers_indices.update(outliers.index.tolist())
        return outliers

    def remove_outliers(self):
        """
        Удаляет строки, содержащие выбросы, из DataFrame.

        Возвращает:
        - DataFrame без выбросов.
        """
        data_cleaned = self.df.drop(index=self.outliers_indices)
        return data_cleaned

    def visualize_outliers(self, feature, tentacle_length=1.5):
        """
        Визуализирует выбросы в указанном признаке с помощью boxplot.

        Параметры:
        - feature: str, имя признака для визуализации выбросов.
        - tentacle_length: float, длина усиков в интервале IQR, по умолчанию 1.5.
        """
        sns.boxplot(x=self.df[feature], whis=tentacle_length)
        plt.title(f'Boxplot of {feature} with whiskers length = {tentacle_length}')
        plt.show()

# Пример использования:
# df - ваш DataFrame
# handler = OutlierHandler(df)
# handler.find_outliers('feature_1')
# handler.find_outliers('feature_2')
# cleaned_data = handler.remove_outliers()
# handler.visualize_outliers('feature_1')

#~=========================================================>

#^ Изоляционный лес (Isolation Forest):

from sklearn.ensemble import IsolationForest

def data_model_clearing(data_x, t_column=None): 
    data = data_x.copy()
    
    # Обучаем изолирующий лес
    isolation_forest = IsolationForest(n_estimators=300, random_state=88111)
    data['estimator'] = isolation_forest.fit_predict(data)
    
    # Отбираем значения -1 (выбросы)
    outliers = data[data['estimator'] == -1]
    print("Количество аномалий: ", len(outliers))
    print("Всего данных: ", len(data))
    
    # Создаем выборку без выбросов
    data_cleaned = data[data['estimator'] != -1].drop(['estimator'], axis=1)
    
    return data_cleaned

# Пример использования
# data_x - ваш DataFrame
# cleaned_data = data_model_clearing(data_x)

