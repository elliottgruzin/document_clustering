import numpy as np
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt


jar = open('sentence_vectors_ng.pkl','rb')
jar2 = open('cluster_to_file.pkl','rb')
sentence_vectors = pickle.load(jar)
cluster_to_file = pickle.load(jar2)

sentences_2d = {}
vectors = []
files = []

for file, vector in sentence_vectors.items():
    files.append(file)
    vectors.append(vector)

two_dim_rep = TSNE().fit_transform(vectors)
x, y = zip(*list(two_dim_rep))

colours= ['r','g','b','c','y']

col = 0

for cluster, clustered_files in cluster_to_file.items():
    if len(clustered_files) < 4:
        continue
    print(col)
    idxs = []
    for file in clustered_files:
        idx = files.index(file)
        idxs.append(idx)
    xs = [x[i] for i in idxs]
    ys = [y[i] for i in idxs]
    plt.scatter(xs, ys, c = colours[col])
    col += 1

plt.savefig('scatter.png')
