import numpy as np
import skfuzzy as fuzz  
import math

cluster_1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
cluster_2 = np.random.normal(loc=[7, 7], scale=0.5, size=(100, 2))
data = np.vstack((cluster_1, cluster_2)).T

def determine_clusters(data):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster(data, n_clusters=3, m=2, error=0.005, maxiter=1000, init=None)
    return 
print(determine_clusters(data))