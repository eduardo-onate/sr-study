import pickle
import sys
import numpy as np

import nemo.collections.asr as nemo_asr
from NeMo.examples.speaker_tasks.recognition.extract_speaker_embeddings import get_embeddings
from utils.operations import cosine_similarity, arccos, shuffle_embeddings, centroid_drift



if len(sys.argv) < 2:
    print("Usage: python process_data.py <manifest_file>")
    sys.exit(1)


    
def analysis():
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    manifest_file = sys.argv[1]
    embeddings_file = get_embeddings(speaker_model, manifest_file, embedding_dir='~/data/')
    data = pickle.load(open(embeddings_file, 'rb'))

    # Store embeddings grouped by label
    embeddings_by_label = {}
    embeddings = []
    labels = []
    # Iterate over the dictionary items
    for key, value in data.items():
        # Parse the key to extract the label
        _, label, _ = key.split('@')

        # If the label is not in the dictionary, add it with an empty list
        if label not in embeddings_by_label:
            embeddings_by_label[label] = []

        # Append the embedding to the correct label list
        embeddings_by_label[label].append(value)

        labels.append(label)
        embeddings.append(value)

    embeddings = np.array(embeddings)
    unique_labels = list(set(labels))
    unique_labels.sort()