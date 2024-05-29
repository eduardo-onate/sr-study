import glob
import pickle
import numpy as np

from tools.analyzer import Analyzer
from utils.operations import cosine_similarity, arccos, shuffle_embeddings, centroid_drift


embedding_files = glob.glob('/home/edu/embeddings/*.pkl')
embedding_files.remove('/home/edu/embeddings/all_manifest_embeddings.pkl')
print(embedding_files)

for f in embedding_files:
    data = pickle.load(open(f, 'rb'))
    label_vector, embeddings, embeddings_by_label = Analyzer.get_embeddings(f)
    
# def analysis():
    
#     embeddings_file = get_embeddings(speaker_model, manifest_file, embedding_dir='~/data/')
#     data = pickle.load(open(embeddings_file, 'rb'))

#     # Store embeddings grouped by label
#     embeddings_by_label = {}
#     embeddings = []
#     labels = []
#     # Iterate over the dictionary items
#     for key, value in data.items():
#         # Parse the key to extract the label
#         _, label, _ = key.split('@')

#         # If the label is not in the dictionary, add it with an empty list
#         if label not in embeddings_by_label:
#             embeddings_by_label[label] = []

#         # Append the embedding to the correct label list
#         embeddings_by_label[label].append(value)

#         labels.append(label)
#         embeddings.append(value)

#     embeddings = np.array(embeddings)
#     unique_labels = list(set(labels))
#     unique_labels.sort()