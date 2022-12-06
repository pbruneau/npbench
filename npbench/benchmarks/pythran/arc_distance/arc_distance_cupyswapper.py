import numpy as np
import cupy as cp


def arc_distance(theta_1, phi_1, theta_2, phi_2):
    if isinstance(phi_2, np.ndarray):
        phi_2 = cp.asarray(phi_2)
    if isinstance(theta_2, np.ndarray):
        theta_2 = cp.asarray(theta_2)
    if isinstance(phi_1, np.ndarray):
        phi_1 = cp.asarray(phi_1)
    if isinstance(theta_1, np.ndarray):
        theta_1 = cp.asarray(theta_1)
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = cp.sin((theta_2 - theta_1) / 2) ** 2 + cp.cos(cp.asarray(theta_1)
        ) * cp.cos(cp.asarray(theta_2)) * cp.sin((phi_2 - phi_1) / 2) ** 2
    distance_matrix = 2 * cp.arctan2(cp.sqrt(temp), cp.sqrt(1 - temp))
    return distance_matrix
