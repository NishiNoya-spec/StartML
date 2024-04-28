from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class TSNEResultVisualizer:
    """
    Класс для визуализации и понижения пространства (t-SNE)
    """
        
    def __init__(self, X, Y, n_components, random_state=42):
        self.X = X
        self.Y = Y
        self.n_components = n_components
        self.random_state = random_state
        self.fit_transform() 

    def fit_transform(self):
        self.tsne = TSNE(n_components=self.n_components, random_state=self.random_state)
        TSNE_dataset = self.tsne.fit_transform(self.X)
        self.TSNE_dataset = pd.DataFrame(TSNE_dataset, columns=[f'TSNE_{i+1}' for i in range(self.n_components)])
        return self.TSNE_dataset

    def plot_TSNE(self):
        if self.n_components == 2:
            fig, ax = plt.subplots(figsize=(16, 10))
            for label in self.Y.unique():
                subset = self.TSNE_dataset[self.Y == label]
                ax.scatter(subset['TSNE_1'], subset['TSNE_2'], label=label)
            ax.legend(title=self.Y.name)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            plt.show()

        elif self.n_components == 3:
            TSNE_dataset_3d = pd.concat((self.TSNE_dataset, self.Y), axis=1)
            fig = plt.figure(figsize=(16, 14))
            ax = fig.add_subplot(111, projection='3d')

            unique_labels = self.Y.unique()
            colors = sns.color_palette('hsv', len(unique_labels))

            for label, color in zip(unique_labels, colors):
                subset = TSNE_dataset_3d[TSNE_dataset_3d[self.Y.name] == label]
                ax.scatter3D(subset['TSNE_1'], 
                             subset['TSNE_2'], 
                             subset['TSNE_3'],
                             c=[color],
                             label=label)

            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            ax.legend(loc='best', title=self.Y.name)
            plt.show()

        else:
            print("Unsupported number of t-SNE components for visualization.")


#~ tsne = TSNEResultVisualizer(X, Y, n_components=2)
#~ tsne.plot_TSNE()

#! ==============================================================>>