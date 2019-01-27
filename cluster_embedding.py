import pickle
from sklearn.cluster import KMeans
import numpy as np
import utils
import pandas as pd
import os

def get_index_from_name(names, name):
    return names.index(name)

#Load the embedded images
names , embeddings = pickle.load(open("embedded_images.p", "rb"))
positive_names = os.listdir("positives_2")
negative_names = os.listdir("negatives_2")
positive_inds = [None for _ in positive_names]
negative_inds = [None for _ in negative_names]
for i,name in enumerate(positive_names):
    positive_inds[i] = get_index_from_name(names, name)
for i,name in enumerate(negative_names):
    negative_inds[i] = get_index_from_name(names, name)

positive_embeddings = embeddings[positive_inds]
#Perform K-mean clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(positive_embeddings)

positives = list(np.argwhere(kmeans.labels_ == 0).astype(np.int))
negatives = list(np.argwhere(kmeans.labels_ == 1).astype(np.int))

# for i in positives:
#     name = positive_names[i[0]]
#     utils.save_image("positives_2\\"+name, "positives_3\\"+name)
# for i in negatives:
#     name = positive_names[i[0]]
#     utils.save_image("positives_2\\"+name, "negatives_3\\"+name)
