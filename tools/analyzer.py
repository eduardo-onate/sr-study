import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from typing import List

from utils.operations import centroid_evolution_by_label


class Analyzer:

    @staticmethod
    def get_embeddings(file_path: str):
        """
        Reads embeddings from a pickle file and returns the label vector, embeddings
        and embeddings grouped by label.
        """
        data = pickle.load(open(file_path, 'rb'))

        # Store embeddings grouped by label
        embeddings_by_label = {}
        embeddings = []
        label_vector = []

        # Iterate over the dictionary items
        for key, value in data.items():
            # Parse the key to extract the label
            _, label, _ = key.split('@')

            # If the label is not in the dictionary, add it with an empty list
            if label not in embeddings_by_label:
                embeddings_by_label[label] = []

            # Append the embedding to the correct label list
            embeddings_by_label[label].append(value)

            label_vector.append(label)
            embeddings.append(value)


        return np.array(label_vector), np.array(embeddings), embeddings_by_label
    

    @staticmethod
    def get_centroid_by_label(embeddings_by_label: dict):
        
        centroids_by_label = {}

        # Calculate centroid for each label
        for label, emb_list in embeddings_by_label.items():
            # Convert list of arrays into a single numpy array
            emb_array = np.array(emb_list)
            # Calculate the mean along axis 0 (mean of each column)
            centroid = np.mean(emb_array, axis=0)
            centroids_by_label[label] = centroid

        return centroids_by_label
    

    @staticmethod
    def centroid_evolution(label_vector: np.ndarray, embeddings: np.ndarray, K: int):
        labels = np.sort(np.unique(label_vector))
        sims_df = {}
        centroids = np.empty((len(labels), K, embeddings.shape[1]), dtype=embeddings.dtype)

        for label_num in labels:
            centroid_sequence, similarities = centroid_evolution_by_label(embeddings, label_vector, label_num)
            sims_df[f'{labels[label_num]}_mean'] = np.mean(similarities, axis=0).flatten()
            sims_df[f'{labels[label_num]}_std'] = np.std(similarities, axis=0).flatten()
            avg_centroid_sequence = np.mean(centroid_sequence, axis=0)
            centroids[label_num] = avg_centroid_sequence

        return centroids, pd.DataFrame(sims_df)
    

    @staticmethod
    def plot_centroid_evolution_by_label(fig: matplotlib.figure.Figure, 
                                         sims_df: pd.DataFrame, label: str | List[str], T: int = 10):
        
        idx = np.arange(2, len(sims_df) + 2)
        ax = fig.add_subplot(1, 1, 1)  # Add an Axes to the figure
        cmap = get_cmap('tab20')  # Get the tab20 colormap
        colors = [cmap(i) for i in range(cmap.N)]  # Get all colors from the colormap
    
        if isinstance(label, str):
            ax.errorbar(idx[:T], sims_df[f'{label}_mean'][:T], 
                        yerr=sims_df[f'{label}_std'][:T], 
                        label='Similitud Coseno Normalizada', 
                        fmt='-', 
                        color=colors[0], 
                        ecolor=colors[10], 
                        elinewidth=2, 
                        capsize=4)
            
        elif isinstance(label, list):
            for i, l in enumerate(label):
                color_mean = colors[i % 10]        # Darker color
                color_std = colors[(i % 10) + 10]  # Lighter color
                ax.errorbar(idx[:T], sims_df[f'{l}_mean'][:T], 
                            yerr=sims_df[f'{l}_std'][:T], 
                            label=f'Similitud Coseno Normalizada {l}', 
                            fmt='-', 
                            color=color_mean, 
                            ecolor=color_std, 
                            elinewidth=2, 
                            capsize=4)

        ax.set_xlabel('Cantidad de audios')
        ax.set_ylabel('Similitud coseno normalizada')
        ax.set_title(f'Similitud coseno entre centroides con respecto al anterior\n"{label}"')
        ax.legend()

        plt.show()

        