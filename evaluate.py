import model_utils
import utils
import numpy as np
import pickle
from keras import Model
import os
import pandas as pd

# Calculate top-k mean averaged precision
def calculate_map(true_ids, predicted_ids):
    # Sum averaged precision over all samples
    mean_averaged_precision = 0
    num_samples = len(true_ids)
    k = len(predicted_ids[0])
    for i in range(num_samples):
        tp = 0
        fp = 0
        # Calculate averaged precision
        true_id = true_ids[i]
        averaged_precision = 0
        for j in range(k):
            pred_id = predicted_ids[i, j]
            tp += true_id == pred_id
            fp += true_id != pred_id
            precision_at_j = tp/(tp+fp)
            recall_change_at_j = 0
            if true_id == pred_id:
                recall_change_at_j = 1/k
            averaged_precision += precision_at_j * recall_change_at_j
        mean_averaged_precision += averaged_precision/num_samples
    return mean_averaged_precision

def get_ids_from_img_names(names, annotations):
    names = list(annotations.loc[names].values[:, 0])
    return names

def generate_embedding(samples, model):
    preds = model.predict(samples)
    pickle.dump(preds, open("embedding32.p", "wb"))

def generate_similarity_matrix():
    # Get model
    original_model = model_utils.identification_model([224, 224, 3], 2)
    if not os.path.exists("embedding32.p"):
        # Get data
        data, annotations, view_predictions = utils.load_all_data()
        # Get a new model from only up to embedding layer
        inp = [original_model.input[0], original_model.input[2]]
        layer = original_model.layers[-4]
        model = Model(inputs=inp, outputs=layer.output)
        generate_embedding([data, view_predictions], model, layer)

    # Starting from distance layer
    distance_layer = original_model.layers[-2]
    model = Model(inputs=distance_layer.input, outputs=original_model.output)
    embedding32 = pickle.load(open("embedding32.p", "rb"))
    train_portion = 0.7
    num_train = int(len(embedding32) * train_portion)
    num_test = len(embedding32) - num_train
    similairty_matrix = np.zeros([num_test, num_train])

    train_samples = embedding32[:num_train]
    test_samples = embedding32[num_train:]

    for i in range(num_test):
        test_sample_repeated = np.reshpae(np.tile(test_samples[i], num_train), [num_train, 32])
        similairty_matrix[i] = model.predict([test_sample_repeated, train_samples])
    pickle.dump(similairty_matrix, open("similarity_matrix.p", "wb"))

# Evaluate
if __name__ == "__main__":
    if not os.path.exists("similarity_matrix.p"):
        generate_similarity_matrix()
    names = pickle.load(open("names.p", "rb"))
    annotations = pd.read_csv("train.csv", index_col=False)
    annotations.set_index("Image", inplace=True)
    ids = get_ids_from_img_names(names, annotations)
    similarity_matrix = pickle.load(open("similarity_matrix.p", "rb"))
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    k = 5
    predicted_ids = [map(lambda idx: ids[idx], row[0:k]) for row in sorted_indices]
    mean_averaged_precision = calculate_map(ids[similarity_matrix.shape[1]:], predicted_ids)
    print("Mean averaged precision:", mean_averaged_precision)
    