import numpy as np

def get_loss(X_one_hot, y, beta0, betas):
    if len(beta0) == 0:
        return 0
    mean_loss = 0
    for i in range(len(betas)):
        wi = betas[i, :]
        intercepti = beta0[i]
        logit = X_one_hot @ wi + intercepti
        y_pred = np.exp(logit) / (1 + np.exp(logit))
        y_pred = np.where(y_pred > 0.5, 1, -1)
        mean_loss += (y != y_pred).mean()
        # print(np.nonzero(wi)[0], (y != y_pred).mean())
    return mean_loss / len(betas)

def average_pairwise_diversity(betas, diversity_metric, limit):
    if len(betas) < 2:
        return 0.0
    num_samples = 1
    if len(betas) > limit:
        num_samples = 1 + int(len(betas) / limit)
    else:
        limit = len(betas)
    
    all_diversities = []
    for _ in range(num_samples):
        diversity = []
        sampled_idx = np.random.choice(len(betas), limit, replace=False)
        sampled_betas = betas[sampled_idx]
        for i in range(len(sampled_betas)):
            for j in range(i + 1, len(sampled_betas)):
                diversity.append(diversity_metric(sampled_betas[i], sampled_betas[j]))
        all_diversities.append(sum(diversity) / len(diversity))
    return sum(all_diversities) / len(all_diversities)

def intersection_over_union(betas_1, betas_2):
    indices_1 = betas_1.nonzero()[0]
    indices_2 = betas_2.nonzero()[0]

    intersection = len(set(indices_1).intersection(set(indices_2)))
    union = len(set(indices_1).union(set(indices_2)))

    return intersection / union

# def correlation(betas_1, betas_2):
#     indices_1 = betas_1.nonzero()[0]
#     indices_2 = betas_2.nonzero()[0]

#     X_subset_1 = X[:, indices_1]
#     X_subset_2 = X[:, indices_2]
        
#     correlation_matrix = np.corrcoef(X_subset_1.T, X_subset_2.T)
#     return np.mean(correlation_matrix)

def euclidean_distance(betas_1, betas_2):
    return np.linalg.norm(betas_1 - betas_2)

def cosine_similarity(betas_1, betas_2):
    dot_product = np.dot(betas_1, betas_2)
    norm_a = np.linalg.norm(betas_1)
    norm_b = np.linalg.norm(betas_2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)