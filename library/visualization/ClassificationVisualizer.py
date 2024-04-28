import seaborn as sns
import matplotlib.pyplot as plt

class NumericBoxPlot:
    def __init__(self, df, target_column):
        """
        ### Box-plot-ы распределений в разных классах

        Параметры:
        - df: DataFrame, исходные данные.
        - target_column: str, имя целевого признака.
        """
        self.df = df
        self.target_column = target_column

    def plot(self):
        """
        Визуализирует числовой признак относительно целевого признака с помощью boxplot.
        """
        numeric_columns = [col for col in self.df.columns if self.df[col].dtype != 'object']
        for col in numeric_columns:
            if col != self.target_column:
                fig = plt.figure()
                fig.set_size_inches(10, 8)
                sns.boxplot(y=self.df[col], x=self.df[self.target_column],
                            data=self.df, hue=self.target_column,
                            palette='Set2', legend=True)
                plt.grid(True, alpha=0.2, linestyle="--")
                plt.show()

### Пример использования

# Создаем экземпляр класса NumericBoxPlot
# box_plotter = NumericBoxPlot(df, 'target_column')

# Используем метод plot для визуализации категориального признака относительно целевого признака
# box_plotter.plot()

### *********************

#~ from library.visualization.ClassificationVisualizer import NumericBoxPlot

#~ box_plotter = NumericBoxPlot(df, 'Segmentation')

#~ box_plotter.plot()

### *********************

#! ==============================================================>>

class CategoricalBarPlot:
    def __init__(self, df, target_column):
        """
        ### Гистограммы распределений в разных классах

        Параметры:
        - df: DataFrame, исходные данные.
        - target_column: str, имя целевого признака.
        """
        self.df = df
        self.target_column = target_column

    def plot(self):
        """
        Строит гистограммы для категориальных признаков относительно целевого признака.
        """
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in categorical_columns:
            if col != self.target_column:
                # Строим баровую диаграмму
                g = sns.catplot(x=col,
                                kind='count',
                                col=self.target_column,
                                hue=col,
                                data=self.df,
                                palette='Set2',
                                sharey=False,
                                stat='count') 
                g.set_xticklabels(rotation=60) 
                plt.show() 

### Пример использования

# Создаем экземпляр класса CategoricalBarPlot
# bar_plotter = CategoricalBarPlot(df, 'target_column')

# Используем метод plot для построения баровых диаграмм для всех категориальных признаков
# bar_plotter.plot()

### *********************

#~ from library.visualization.ClassificationVisualizer import NumericBoxPlot

#~ box_plotter = NumericBoxPlot(df, 'Segmentation')

#~ box_plotter.plot()

### *********************

#! ==============================================================>>
