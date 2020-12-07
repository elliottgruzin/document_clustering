import math
import pickle
import pprint
import numpy as np

jar = open('sentence_vectors_ng.pkl','rb')
email_vectors = pickle.load(jar)
# jar2 = open('sentence_vectors_bert.pkl','rb')
# points = pickle.load(jar2)

files = []
points = []

for key, value in email_vectors.items():
    files.append(key)
    points.append(value)

points_copy = points.copy()

# define radius of kernel

kernel_size = 0.001

# find cluster for each point copy

cluster_centres = []

for point in points_copy:
    is_stable = False
    current_point = point
    while is_stable == False:
        # print('yo')
        update = np.zeros(120)
        n = 0
        for original_point in points:
            if np.linalg.norm(current_point - original_point) < 2.25:
                update += original_point
                n += 1
        new_point = update/n
        if (new_point == current_point).all():
            is_stable = True
        current_point = new_point
    cluster_centres.append(current_point)


cluster_bank = []
cluster_names = []
cluster_to_file = {}
cluster_counter = 0

for i, centroid in enumerate(cluster_centres):
    if not any((centroid == x).all() for x in cluster_bank):
        cluster_counter += 1
        cluster_name = 'cluster_'+str(cluster_counter)
        cluster_bank.append(centroid)
        cluster_names.append(cluster_name)
        cluster_to_file[cluster_name] = [files[i]]
    else:
        lst = [(centroid == x).all() for x in cluster_bank]
        idx = lst.index(True)
        cluster_name = cluster_names[idx]
        cluster_to_file[cluster_name].append(files[i])
# pprint.pprint(sentence_vectors)
n =0

for cluster, files in cluster_to_file.items():
    if len(files) > 1:
        n+=1

with open('cluster_to_file.pkl','wb') as output:
    pickle.dump(cluster_to_file, output)
# print(n)
pprint.pprint(cluster_to_file)
