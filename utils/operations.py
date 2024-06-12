import numpy as np
import pandas as pd

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return (cosine_similarity + 1) / 2  # Normalized cosine similarity

def arccos(x):
    """
    Compute the arccosine of a value in radians.
    """
    cos_theta = 2*x - 1  # De-normalize cosine similarity
    return np.arccos(cos_theta)


def shuffle_embeddings(embeddings_by_label: dict, label: str, num_shuffles: int = 100,
                   embeddings_per_shuffle: int = 50) -> np.ndarray:
    """
    Pseudo-randomly shuffles embeddings for a given label.
    """
    subject_embeddings = np.array(embeddings_by_label[label])
    if embeddings_per_shuffle > subject_embeddings.shape[0]:
        return None
    base_seed = hash(label) & 0xffffffff  # Use `& 0xffffffff` for a 32-bit integer
    shuffled_embeddings = np.empty((num_shuffles, embeddings_per_shuffle, subject_embeddings.shape[1]), 
                                   dtype=subject_embeddings.dtype)

    for i in range(num_shuffles):
        seed = base_seed + i
        np.random.seed(seed)
        permuted_indices = np.random.permutation(subject_embeddings.shape[0])[:embeddings_per_shuffle]
        shuffled_embeddings[i] = subject_embeddings[permuted_indices]

    return shuffled_embeddings

def centroid_evolution_by_label(embeddings: np.ndarray, labels: np.ndarray, label: str):
    """
    Characterizes the centroid drift for a given label.
    The centroid drift is calculated as the cosine similarity between the centroid of the first n embeddings
    and the centroid of the first n+1 embeddings.
    We use 100 sequences of the embeddings reordenated randomly to avoid any bias.
    """
    shuffled_embeddings = shuffle_embeddings(embeddings, labels, label, num_shuffles=100, embeddings_per_shuffle=K)
    centroids = np.empty(shuffled_embeddings.shape, dtype=shuffled_embeddings.dtype)

    for i in range(shuffled_embeddings.shape[0]):
        embeddings_sequence = shuffled_embeddings[i]
        for j in range(embeddings_sequence.shape[0]):
            embeddings = embeddings_sequence[:j+1]
            cen = np.mean(embeddings, axis=0)
            centroids[i, j] = cen
        
    similarities = np.empty((shuffled_embeddings.shape[0], shuffled_embeddings.shape[1] - 1, 1))
    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]):
            cen1 = centroids[i, j]
            cen2 = centroids[i, j+1]
            sim = cosine_similarity(cen1, cen2)
            similarities[i, j] = sim

    return centroids, similarities
