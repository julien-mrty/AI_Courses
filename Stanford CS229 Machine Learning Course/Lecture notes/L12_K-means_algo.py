import numpy as np


"""
IV) Unsupervised learning
Chapter 10
Clustering and the k-means algorithm
"""


# np.random.seed(0)  # For reproducibility


def k_means_plus_plus_init(input_feature, num_clusters):
    n_sample = len(input_feature)

    # Choose the first center randomly
    clusters_centers = np.zeros(num_clusters)
    clusters_centers[0] = input_feature[np.random.randint(n_sample)]

    for i in range(1, num_clusters):
        # Compute distances from each data point to the nearest existing cluster center
        distances = np.abs(input_feature[:, np.newaxis] - clusters_centers[:i])
        # Only keep the minimal distances. Keeping maxiclusters_centersm distances could lead to chose a point which is far from one
        # cluster but close from another one
        min_distances = np.min(distances, axis=1)

        # Compute probabilities proportional to the squared distances
        # Among the minimal distances, we assign the highest probability to be chosen at the biggest distance
        probabilities = min_distances ** 2
        probabilities /= np.sum(probabilities)

        # Choose the next center with probability proportional to the squared distance
        # Choosing based on a probability distribution allow to choose different clusters_centers when you rerun the algorithm
        # (non-deterministic). This can avoid some suboptimal clusters_centers initialization. This ensures diversity in clusters_centers placements
        clusters_centers[i] = np.random.choice(input_feature, p=probabilities)

    return clusters_centers


def k_means(input_feature, num_clusters, cluster_tolerance, distortion_tolerance, max_iters=100):

    clusters_centers = k_means_plus_plus_init(input_feature, num_clusters)

    # Initialize the cluster here to be able to return it
    clusters_assignments = np.zeros(len(input_feature), dtype=int)

    prev_distortion = np.inf
    prev_clusters_centers = np.array(clusters_centers)

    for iteration in range(max_iters):

        # Compute distances from each data point to each cluster center
        distances = np.abs(input_feature[:, np.newaxis] - clusters_centers)
        # Assign the nearest cluster center to each point
        clusters_assignments = np.argmin(distances, axis=1)

        # Update cluster centers
        for j in range(num_clusters):
            points_in_cluster = input_feature[clusters_assignments == j]

            # If no points are assigned to this cluster, keep the old mean (could also randomize)
            if len(points_in_cluster) > 0:
                clusters_centers[j] = np.mean(points_in_cluster)
            else:
                clusters_centers[j] = np.random.choice(input_feature)

        # Check for convergence with distortion
        distortion = distortion_function(input_feature, clusters_centers, clusters_assignments)
        if np.abs(distortion - prev_distortion) < distortion_tolerance:
            print(f"Converged early in {iteration + 1} iterations, MU = {clusters_centers}, Distortion = {distortion}")
            break

        prev_distortion = distortion

        # Check for convergence with clusters center movement
        cluster_dif = np.linalg.norm(clusters_centers - prev_clusters_centers)
        if cluster_dif < cluster_tolerance:
            print(f"Converged in {iteration + 1} iterations, clusters_centers = {clusters_centers}, Difference = {cluster_dif}, Distortion = {distortion}")
            break

        prev_clusters_centers = clusters_centers

        print(f"Iteration {iteration + 1}: clusters_centers = {clusters_centers}, Difference = {cluster_dif}, Distortion = {distortion}")

    return clusters_centers, clusters_assignments


def distortion_function(input_feature, clusters_centers, clusters_assignments):
    J = np.abs(input_feature - clusters_centers[clusters_assignments])
    return np.sum(J)


def main():
    """ Samples """
    num_clusters = 3
    input_1 = np.random.normal(loc=1, scale=1.5, size=10)
    input_2 = np.random.normal(loc=5, scale=1, size=30)
    input_3 = np.random.normal(loc=8, scale=1.2, size=20)
    input_feature = np.concatenate((input_1, input_2, input_3), axis=0)

    k_means(input_feature, num_clusters, 1e-300, 1e-300)


if __name__ == "__main__":
    main()
