from dtw import *
import numpy as np
import matplotlib.pyplot as plt

path1 = np.array([
    [0,0],
    [1,1],
    [2,2],
    [3,3]
])

path2 = np.array([
    [0,0],
    [1.1,1],
    [2.1,2],
    [3,3]
])

alignment = dtw(path1, path2)

print(alignment.distance)

