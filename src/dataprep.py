import numpy as np
import torch

# Target function to approximate
def target_function(x):
    return (
        np.sin(3 * x[:, 0]) + np.sin(3 * x[:, 1]) + np.sin(3 * x[:, 2]) - 2
    ) ** 2 * np.cos(7 * x[:, 3])


# random gaussian vector of dim d
def generate_data(N, d=4, seed=42):
    np.random.seed(seed)
    x = np.random.randn(N, d)  # Generate Gaussian features
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # Normalize
    y = target_function(x)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).view(-1, 1)