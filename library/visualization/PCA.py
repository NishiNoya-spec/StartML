import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

class PCAResultVisualizer:

    """
    Класс для визуализации и понижения пространства (PCA)
    """
        
    def __init__(self, X, Y, n_components):
        self.X = X.subtract(X.mean())  # Центрируем данные
        self.Y = Y
        self.n_components = n_components
        self.fit_transform() 

    def fit_transform(self):
        self.pca = PCA(n_components=self.n_components)
        PCA_dataset = self.pca.fit_transform(self.X)
        self.PCA_dataset = pd.DataFrame(PCA_dataset, columns=[f'PCA_{i+1}' for i in range(self.n_components)])
        return self.PCA_dataset

    def plot_correlation_heatmap(self):
        corrs = pd.DataFrame()
        for i in range(self.n_components):
            component_corr = self.X.corrwith(self.PCA_dataset[f'PCA_{i+1}'])
            corrs[f'PCA_{i+1}'] = component_corr

        corrs.columns = [f'PCA_{i+1}' for i in range(self.n_components)]
        fig = plt.figure(figsize=(16, 12))
        hm = sns.heatmap(corrs, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=corrs.index, xticklabels=corrs.columns)
        plt.title('Features Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_PCA(self):

        if self.n_components == 2:

            fig, ax = plt.subplots(figsize=(16, 10))
            for label in self.Y.unique():
                subset = self.PCA_dataset[self.Y == label]
                ax.scatter(subset['PCA_1'], subset['PCA_2'], label=label)
            ax.legend(title=self.Y.name)
            ax.set_xlabel('PCA_1')
            ax.set_ylabel('PCA_2')

        elif self.n_components == 3:
            
            PCA_dataset_3d = pd.concat((self.PCA_dataset, self.Y), axis=1)
            fig = plt.figure(figsize=(16, 14))
            ax = fig.add_subplot(111, projection='3d')

            unique_labels = self.Y.unique()
            colors = sns.color_palette('hsv', len(unique_labels))

            for label, color in zip(unique_labels, colors):
                subset = PCA_dataset_3d[PCA_dataset_3d[self.Y.name] == label]
                ax.scatter3D(subset['PCA_1'], 
                                subset['PCA_2'], 
                                subset['PCA_3'],
                                c=[color],
                                label=label)

            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')

            ax.legend(loc='best', title=self.Y.name)
            plt.show()

        else:
            print("Unsupported number of PCA components for visualization.")



#~ pcv = PCAResultVisualizer(X, Y, n_components=2)
#~ pcv.plot_correlation_heatmap()
#~ pcv.plot_PCA()

#! ==============================================================>>