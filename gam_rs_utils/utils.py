import numpy as np

def get_loss(X_one_hot, y, beta0, betas):
    mean_loss = 0
    for i in range(len(betas)):
        wi = betas[i, :]
        intercepti = beta0[i]
        logit = X_one_hot @ wi + intercepti
        y_pred = np.exp(logit) / (1 + np.exp(logit))
        y_pred = np.where(y_pred > 0.5, 1, -1)
        mean_loss += (y != y_pred).mean()
        print(np.nonzero(wi)[0], (y != y_pred).mean())
    return mean_loss / len(betas)

def average_pairwise_diversity(betas, diversity_metric):
    diversity = []
    for i in range(len(betas)):
        for j in range(i, len(betas)):
            diversity.append(diversity_metric(betas[i], betas[j]))
    return sum(diversity) / len(diversity)

def intersection_over_union(betas_1, betas_2):
    indices_1 = betas_1.nonzero()[0]
    indices_2 = betas_2.nonzero()[0]

    intersection = len(set(indices_1).intersection(set(indices_2)))
    union = len(set(indices_1).union(set(indices_2)))

    return intersection / union

def correlation(self, betas_1, betas_2):
    indices_1 = betas_1.nonzero()[0]
    indices_2 = betas_2.nonzero()[0]

    X_subset_1 = self.X[:, indices_1]
    X_subset_2 = self.X[:, indices_2]
        
    correlation_matrix = np.corrcoef(X_subset_1.T, X_subset_2.T)
    return np.mean(correlation_matrix)