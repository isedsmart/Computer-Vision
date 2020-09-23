#%matplotlib inline
#%matplotlib notebook
##%load_ext autoreload
#%autoreload 2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *
from student_harris import get_interest_points
#from IPython.core.debugger import set_trace

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.
u
    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    compare_distances = []
    confidences = []
    f1matches = []
    f2matches = []
    threshold_val = 0.95
    for f1_sect, f1_index in enumerate(features1):
        distances = []
        for f2_sect, f2_index in enumerate(features2):
            for value in range(len(features1[f1_sect])):
                dist = np.linalg.norm(f2_index[f2_sect] - f1_index[f1_sect])
                to_append = (dist, f2_sect, f1_sect)
            distances.append(to_append)
        sorted_distances = sorted(distances, key=lambda tup: tup[0])
        shortestDist, f1i, f2i = sorted_distances[0]
        secShortestDist, secf1i, secf2i = sorted_distances[1]
        confidence = shortestDist/secShortestDist
        # setting up the threshold
        if (confidence < threshold_val):
            confidences.append(confidence)
            f1matches.append(f1i)
            f2matches.append(f2i)

    confidences = np.asarray(confidences)
    matches = np.column_stack((f1matches,f2matches))

    return matches, confidences

