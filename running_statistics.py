# type: ignore
"""
See code at the end of the file for example usage.

"""

import torch

class RunningStats:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shape=(), dtype=torch.float64, device="cuda"):
        """Tracks the mean, variance and count of values."""
        # shape is a tuple of two integers
        assert len(shape) == 2

        self.epsilon = epsilon
        self.shape = shape
        self.dtype = dtype
        self.device = device

        self.data = {
            "count": epsilon,
            "num_nonzero_x": torch.zeros(shape[0], dtype=torch.int32, device=device),
            "num_nonzero_y": torch.zeros(shape[1], dtype=torch.int32, device=device),
            "max_x": torch.zeros(shape[0], dtype=dtype, device=device),
            "max_y": torch.zeros(shape[1], dtype=dtype, device=device),
            "mean_x": torch.zeros(shape[0], dtype=dtype, device=device),
            "mean_y": torch.zeros(shape[1], dtype=dtype, device=device),
            "var_x": torch.ones(shape[0], dtype=dtype, device=device),
            "var_y": torch.ones(shape[1], dtype=dtype, device=device),
            "cov": torch.ones(shape, dtype=dtype, device=device),
        }

    def __getattr__(self, name):
        # Redirect attribute access to the data dictionary
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def update(self, x, y):
        """Updates the mean, var and count from a batch of samples."""
        assert x.shape[-1] == self.shape[0] and y.shape[-1] == self.shape[1]
        assert x.ndim > 1 and y.ndim > 1
        if x.ndim > 2:
            x = x.view(-1, self.shape[0])
        if y.ndim > 2:
            y = y.view(-1, self.shape[1])
        assert x.shape[0] == y.shape[0], "We use torch.cov which requires number of samples for x and y to be the same."
        if x.device != self.device:
            x = x.to(self.device)
        if y.device != self.device:
            y = y.to(self.device)

        batch_data = {"count": x.shape[0], 
                      "num_nonzero_x": x.count_nonzero(dim=0),
                      "num_nonzero_y": y.count_nonzero(dim=0),
                      "max_x": torch.amax(x, dim=0),
                      "max_y": torch.amax(y, dim=0),
                      "mean_x":  torch.mean(x, dim=0), 
                      "mean_y": torch.mean(y, dim=0), 
                      "var_x": torch.var(x, dim=0),
                      "var_y": torch.var(y, dim=0), 
                      "cov": torch.cov(torch.cat((x, y), dim=1).t())[:self.shape[0], self.shape[0]:]}
        
        self._check_shapes(batch_data) # a sanity check
        self._update_data(batch_data)

    def _update_data(self, batch_data):
        """Updates from batch mean, variance and count moments."""

        # update num_nonzero
        num_nonzero_x = self.num_nonzero_x + batch_data["num_nonzero_x"]
        num_nonzero_y = self.num_nonzero_y + batch_data["num_nonzero_y"]

        # update max
        max_x = torch.maximum(self.max_x, batch_data["max_x"])
        max_y = torch.maximum(self.max_y, batch_data["max_y"])

        # update mean
        delta_x = batch_data["mean_x"] - self.mean_x # (n, )
        delta_y = batch_data["mean_y"] - self.mean_y # (m, )
        tot_count = self.count + batch_data["count"]

        new_mean_x = self.mean_x + delta_x * batch_data["count"] / tot_count # (n, )
        new_mean_y = self.mean_y + delta_y * batch_data["count"] / tot_count # (m, )

        # update var
        new_var_x = update_var(self.var_x, self.count, batch_data["var_x"], batch_data["count"], delta_x, tot_count)
        new_var_y = update_var(self.var_y, self.count, batch_data["var_y"], batch_data["count"], delta_y, tot_count)

        # update cov
        new_cov = update_cov(self.cov, self.count, batch_data["cov"], batch_data["count"], delta_x, delta_y, tot_count)

        self.data = {
            "max_x": max_x,
            "max_y": max_y,
            "mean_x": new_mean_x,
            "mean_y": new_mean_y,
            "var_x": new_var_x,
            "var_y": new_var_y,
            "cov": new_cov,
            "count": tot_count,
            "num_nonzero_x": num_nonzero_x,
            "num_nonzero_y": num_nonzero_y,
        }

    @property
    def corr_matrix(self):
        # May be called after all updates are done.
        corr = self.cov / (torch.outer(torch.sqrt(self.var_x), torch.sqrt(self.var_y)))
        corr_matrix = torch.where(torch.isnan(corr), torch.tensor(-2.0), corr)
        return corr_matrix
    
    @property
    def density_x(self):
        return self.num_nonzero_x / self.count
    
    @property
    def density_y(self):
        return self.num_nonzero_y / self.count
    
    def _check_shapes(self, batch_data):
        assert batch_data["mean_x"].shape == (self.shape[0],)
        assert batch_data["mean_y"].shape == (self.shape[1],)
        assert batch_data["var_x"].shape == (self.shape[0],)
        assert batch_data["var_y"].shape == (self.shape[1],)
        assert batch_data["cov"].shape == self.shape, f"expected shape {self.shape}, but got {batch_data['cov'].shape}"

    def to_cpu(self):
        """Returns a copy of the instance with all tensors moved to CPU."""
        # Create a new instance with the same shape and dtype
        cpu_copy = RunningStats(
            epsilon=self.epsilon, 
            shape=self.shape, 
            dtype=self.dtype,  # Match dtype
            device="cpu"          # Target the CPU
        )
        
        # Copy all tensors to the CPU
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                cpu_copy.data[key] = value.to("cpu")
            else:
                cpu_copy.data[key] = value  # Non-tensor values remain unchanged

        return cpu_copy

def update_var(var, count, batch_var, batch_count, delta, tot_count):
    m_a = var * count # M_{2, A} # (n,)
    m_b = batch_var * batch_count # M_{2, B} #(n, )
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count # (n, )
    new_var = M2 / tot_count 
    return new_var

def update_cov(cov, count, batch_cov, batch_count, delta_x, delta_y, tot_count):
    cov_a = cov * count # (n, m)
    cov_b = batch_cov * batch_count # (n, m)
    _prod = delta_x[:, None] @ delta_y[None, :]
    _num = cov_a + cov_b + _prod * count * batch_count / tot_count # TODO: delta_x * delta_y might need shaping
    new_cov = _num / tot_count 
    return new_cov

##############################################################################
### USAGE

# type: ignore
# import numpy as np
# import torch
# from universality.running_statistics import RunningVarCov

# def generate_symmetric_positive_definite_matrix(n):
#     # Step 1: Create a random n x n matrix
#     A = np.random.rand(n, n)
#     # Step 2: Symmetrize by A * A.T
#     symmetric_matrix = np.dot(A, A.T)
#     # Optionally, you can add `n * I` to ensure positive definiteness
#     symmetric_positive_definite_matrix = symmetric_matrix + n * np.eye(n)    
#     return symmetric_positive_definite_matrix


# def get_corr_matrix(cov_matrix: np.ndarray):
#     # Calculate the standard deviations (square root of diagonal elements)
#     std_dev = np.sqrt(np.diag(cov_matrix))
#     # Calculate the outer product of the standard deviations
#     outer_std_dev = np.outer(std_dev, std_dev)
#     # Divide the covariance matrix by the outer product of the standard deviations
#     correlation_matrix = cov_matrix / outer_std_dev
#     # To handle any numerical precision issues, you can set the diagonal to 1
#     # np.fill_diagonal(correlation_matrix, 1.0)
#     return correlation_matrix

# dim_a = 10
# split_a = int(0.4 * dim_a)
# split_b = dim_a - split_a
# a_mean = np.random.randn(dim_a,)
# a_cov = generate_symmetric_positive_definite_matrix(dim_a)

# running_var_cov = RunningVarCov(shape=(split_a, split_b))
# device = "cuda" if torch.cuda.is_available() else "cpu"
# for _ in range(1000):
#     a = torch.from_numpy(np.random.multivariate_normal(a_mean, a_cov, size=100))
#     a = a.to(device)
#     running_var_cov.update(a[:, :split_a], a[:, split_a:])

# a_corr = get_corr_matrix(a_cov)
# print(a_corr[:split_a, split_a:]) # for correlation between split_a and split_b

# print(running_var_cov.get_corr_matrix()) # should be close to the output above