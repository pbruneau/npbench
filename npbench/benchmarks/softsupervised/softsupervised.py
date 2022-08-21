import numpy as np

def initialize(N, d, prop):
    np.random.seed(42)
    data = np.random.normal(size=(N,d))
    labels = np.random.choice(range(3), size=N)
    random_unlabeled_points = np.random.rand(len(labels)) < prop
    labels[random_unlabeled_points] = -1
    
    return data, labels
