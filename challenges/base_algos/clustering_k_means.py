from sklearn.cluster import KMeans

def algorithm(xs):
    """
    This algorithm implements simple k-means clustering with a fixed arbitrary number of clusters (not data-based).

    :param np.ndarray points with shape (num_points, dims)
    :return:
        cluster indices (num_points,)
    """
    n_clusters = 3  # arbitrarily chosen
    cluster_inds = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(xs) + 1

    return cluster_inds