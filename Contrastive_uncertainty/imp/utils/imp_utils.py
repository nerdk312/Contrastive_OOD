import torch 

# Nawid - calculate distances
def compute_distances(protos, example):
  dist = torch.sum((example - protos)**2, dim=2)
  return dist

# Nawid - calculate entropy from counts
def entropy(counts):
    """Compute entropy from discrete counts"""
    if len(counts.shape) > 1:
        counts = counts.flatten()
    N = np.sum(counts)
    p = counts / float(N)
    p = p[np.nonzero(p)]
    return -np.sum(p*np.log(p))


# Nawid - compute logits of being in a cluster based on euclidean distance
def compute_logits(cluster_centers, data):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
        cluster_centers: [K, D] Cluster center representation.
        data: [ N, D] Data representation.
    Returns:
        log_prob: [N, K] logits.
    """
    cluster_centers = torch.unsqueeze(cluster_centers, 0)  # [1, K, D]
    data = torch.unsqueeze(data, 1)  # [N, 1, D]
    # [N, K, D]
    neg_dist = -torch.sum((data - cluster_centers)**2, 2) # Shape (N,K)
    return neg_dist


# Nawid - assign data to cluster centres using k-means 
def assign_cluster(cluster_centers, data):
    """Assigns data to cluster center, using K-Means.
    Args:
        cluster_centers: [K, D] Cluster center representation.
        data: [N, D] Data representation.
    Returns:
        prob: [N, K] Soft assignment.
    """
    logits = compute_logits(cluster_centers, data)  # [N, K]
    prob = F.softmax(logits, dim=-1) # Probability of each datapoint belonging to a class 
    return prob


def compute_logits_radii(cluster_centers, data, radii, prior_weight=1.):
    """Computes the logits of being in one cluster, squared Euclidean.

    Args:
        cluster_centers: [K, D] Cluster center representation.
        data: [N, D] Data representation.
        radii: [ K] Cluster radii.
    Returns:
        log_prob: [ N, K] logits.
    """
    cluster_centers = torch.unsqueeze(cluster_centers, 0)   # [1, K, D]
    data = torch.unsqueeze(data, 1)  # [ N, 1, D]
    dim = data.size()[-1]
    radii = torch.unsqueeze(radii, 0)  # [1,K]
    neg_dist = -torch.sum((data - cluster_centers)**2, dim=2)   # [N, K]

    logits = neg_dist / 2.0 / (radii)
    norm_constant = 0.5*dim*(torch.log(radii) + np.log(2*np.pi))

    logits = logits - norm_constant
    return logits


# Nawid - assign data to cluster centres using k-means 
def assign_cluster_radii(cluster_centers, data, radii):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [K, D] Cluster center representation.
        data: [N, D] Data representation.
        radii: [K] Cluster radii.
    Returns:
        prob: [N, K] Soft assignment.
    """
    logits = compute_logits_radii(cluster_centers, data, radii) # [N, K]
    prob = F.softmax(logits, dim=-1)
    return prob