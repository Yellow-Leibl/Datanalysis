import numpy as np
from Datanalysis.SamplesTools import median


def get_distance_metric(metric: str):
    if metric == 'euclidean':
        return euclidean_distance
    elif metric == 'manhattan':
        return manhattan_distance
    elif metric == 'chebyshev':
        return chebyshev_distance
    elif metric == 'minkowski':
        return minkowski_distance
    elif metric == 'mahalanobis':
        return mahalanobis_distance
    else:
        raise ValueError('Unknown metric')


def euclidean_distance(x1, x2):
    return minkowski_distance(x1, x2, 2)


def weighted_euclidean_distance(x1, x2, w):
    return np.sqrt(np.sum(w * (x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return minkowski_distance(x1, x2, 1)


def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))


def minkowski_distance(x1, x2, m):
    return np.sum(np.abs(x1 - x2) ** m) ** (1 / m)


def mahalanobis_distance(x1, x2, cov):
    return np.sqrt((x1 - x2) @ np.linalg.inv(cov) @ (x1 - x2).T)


def similarity(x1, x2, d):
    return np.exp(-d(x1, x2))


def nearest_neighbor_distance(s1, s2, d):
    min_dist = np.inf
    for x1 in s1:
        for x2 in s2:
            dist = d(x1, x2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def furthest_neighbor_distance(s1, s2, d):
    max_dist = 0
    for x1 in s1:
        for x2 in s2:
            dist = d(x1, x2)
            if dist > max_dist:
                max_dist = dist
    return max_dist


def weighted_average_distance(s1, s2, d):
    N1 = len(s1)
    N2 = len(s2)
    sum_dist = 0
    for x1 in s1:
        for x2 in s2:
            sum_dist += d(x1, x2)
    return sum_dist / (N1 * N2)


def unweighted_average_distance(s1, s2, d):
    sum_dist = 0
    for x1 in s1:
        for x2 in s2:
            sum_dist += d(x1, x2)
    return sum_dist / 4


def median_distance(s1, s2, d):
    dist = []
    for x1 in s1:
        for x2 in s2:
            dist.append(d(x1, x2) / 2)
    return median(dist)


def centroid_distance(s1, s2, d):
    return d(np.mean(s1, axis=0), np.mean(s2, axis=0))


def wards_distance(s1, s2, d):
    N1 = len(s1)
    N2 = len(s2)
    return (N1 * N2 / (N1 + N2)) * centroid_distance(s1, s2, d) ** 2


def lance_williams_distance(d12, d13, d23, alpha1, alpha2, beta, eps):
    return alpha1 * d13 + alpha2 * d23 + beta * d12 + eps * np.abs(d13 - d23)


def get_distance_method_consts(method: str):
    if method == 'nearest':
        return nearest_neighbor_distance_alpha_beta_eps()
    elif method == 'furthest':
        return furthest_neighbor_distance_alpha_beta_eps()
    elif method == 'average':
        return weighted_average_distance_alpha_beta_eps()
    elif method == 'unweighted':
        return unweighted_average_distance_alpha_beta_eps()
    elif method == 'median':
        return median_distance_alpha_beta_eps()
    elif method == 'centroid':
        return centroid_distance_alpha_beta_eps()
    elif method == 'wards':
        return wards_distance_alpha_beta_eps()
    else:
        raise ValueError('Unknown method')


def nearest_neighbor_distance_alpha_beta_eps():
    return 0.5, 0.5, 0, -0.5


def furthest_neighbor_distance_alpha_beta_eps():
    return 0.5, 0.5, 0, 0.5


def weighted_average_distance_alpha_beta_eps(s1, s2):
    N1 = len(s1)
    N2 = len(s2)
    alpha1 = N1 / (N1 + N2)
    alpha2 = N2 / (N1 + N2)
    return alpha1, alpha2, 0, 0


def unweighted_average_distance_alpha_beta_eps():
    return 0.5, 0.5, 0, 0


def median_distance_alpha_beta_eps():
    return 0.5, 0.5, -0.25, 0


def centroid_distance_alpha_beta_eps(s1, s2):
    N1 = len(s1)
    N2 = len(s2)
    alpha1 = N1 / (N1 + N2)
    alpha2 = N2 / (N1 + N2)
    beta = -alpha1 * alpha2
    return alpha1, alpha2, beta, 0


def wards_distance_alpha_beta_eps(s1, s2, s3):
    N1 = len(s1)
    N2 = len(s2)
    N3 = len(s3)
    alpha1 = (N3 + N1) / (N1 + N2 + N3)
    alpha2 = (N3 + N2) / (N1 + N2 + N3)
    beta = -N3 / (N1 + N2 + N3)
    return alpha1, alpha2, beta, 0
