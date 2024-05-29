from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from utils.operations import cosine_similarity, arccos, shuffle_embeddings, centroid_drift


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
    def 