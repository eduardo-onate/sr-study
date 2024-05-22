import numpy as np
import pandas as pd

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    Since embeddings are already normalized, we can use the dot product as the cosine similarity.
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

def centroid_drift(label: str, plot: bool = True):
    """
    Characterizes the centroid drift for a given label.
    The centroid drift is calculated as the cosine similarity between the centroid of the first n embeddings
    and the centroid of the first n+1 embeddings.
    We use 100 sequences of the embeddings reordenated randomly to avoid any bias.
    """
    shuffled_embeddings = shuffle_embeddings(label)
    if shuffled_embeddings is None:
        return None
    subject_centroids = np.empty(shuffled_embeddings.shape, dtype=shuffled_embeddings.dtype)

    for i in range(shuffled_embeddings.shape[0]):
        embeddings_sequence = shuffled_embeddings[i]
        for j in range(embeddings_sequence.shape[0]):
            embeddings = embeddings_sequence[:j+1]
            centroid = np.mean(embeddings, axis=0)
            subject_centroids[i, j] = centroid
    

    similitudes = np.empty((shuffled_embeddings.shape[0], shuffled_embeddings.shape[1] - 1, 1))
    for i in range(similitudes.shape[0]):
        for j in range(similitudes.shape[1]):
            cen1 = subject_centroids[i, j]
            cen2 = subject_centroids[i, j+1]
            sim = cosine_similarity(cen1, cen2)
            similitudes[i, j] = sim

    mean = np.mean(similitudes, axis=0).flatten()
    std = np.std(similitudes, axis=0).flatten()
    return mean, std
