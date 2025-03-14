import numpy as np

# Example list of losses
losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
inv_losses = 1 / losses
probabilities = np.array(inv_losses) / np.sum(inv_losses)
sampled_indices = np.random.choice(len(losses), size=len(losses), replace=False, p=probabilities)

print("Probabilities:", probabilities)
print("Sampled indices:", sampled_indices)


for idx in sampled_indices:
    print(idx)
print(sampled_indices[0])

# make a 2d array
a = np.array([[1, 2, 3], [4, 5, 6]])
for r in a:
    print(r)