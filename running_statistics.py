# type: ignore
"""
See code at the end of the file for example usage.

"""


import numpy as np


class RunningVarCov:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        # shape is a tuple of two integers
        assert len(shape) == 2

        self.shape = shape

        self.data = {
            "count": epsilon,
            "mean_x": np.zeros(shape[0], dtype=dtype),
            "mean_y": np.zeros(shape[1], dtype=dtype),
            "var_x": np.ones(shape[0], dtype=dtype),
            "var_y": np.ones(shape[1], dtype=dtype),
            "cov": np.ones(shape, dtype=dtype),
        }


    def update(self, x, y):
        """Updates the mean, var and count from a batch of samples."""
        assert x.shape[-1] == self.shape[0] and y.shape[-1] == self.shape[1]
        assert x.ndim > 1 and y.ndim > 1
        if x.ndim > 2:
            x = x.view(-1, self.shape[0])
        if y.ndim > 2:
            y = y.view(-1, self.shape[1])

        assert x.shape[0] == y.shape[0], "We use np.cov which requires number of samples for x and y to be the same."

        batch_data = {"count": x.shape[0], 
                      "mean_x":  np.mean(x, axis=0), 
                      "mean_y": np.mean(y, axis=0), 
                      "var_x": np.var(x, axis=0),
                      "var_y": np.var(y, axis=0), 
                      "cov": np.cov(x, y, rowvar=False)[:self.shape[0], self.shape[0]:]}
        
        self._check_shapes(batch_data)

        self.update_from_moments(batch_data)

    def update_from_moments(self, batch_data):
        """Updates from batch mean, variance and count moments."""
        self.data = update_data_from_moments(
            self.data, batch_data
        )

    def get_corr_matrix(self):
        return self.data["cov"] / np.outer(np.sqrt(self.data["var_x"]), np.sqrt(self.data["var_y"]))

    def _check_shapes(self, batch_data):
        assert batch_data["mean_x"].shape == (self.shape[0],)
        assert batch_data["mean_y"].shape == (self.shape[1],)
        assert batch_data["var_x"].shape == (self.shape[0],)
        assert batch_data["var_y"].shape == (self.shape[1],)
        assert batch_data["cov"].shape == self.shape, f"expected shape {self.shape}, but got {batch_data['cov'].shape}"

def update_var_from_moments(var, count, batch_var, batch_count, delta, tot_count):
    m_a = var * count # M_{2, A} # (n,)
    m_b = batch_var * batch_count # M_{2, B} #(n, )
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count # (n, )
    new_var = M2 / tot_count 
    return new_var

def update_cov_from_moments(cov, count, batch_cov, batch_count, delta_x, delta_y, tot_count):
    cov_a = cov * count # (n, m)
    cov_b = batch_cov * batch_count # (n, m)
    _prod = delta_x[:, None] @ delta_y[None, :]
    _num = cov_a + cov_b + _prod * count * batch_count / tot_count # TODO: delta_x * delta_y might need shaping
    new_cov = _num / tot_count 
    return new_cov

def update_data_from_moments(
    data, batch_data
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta_x = batch_data["mean_x"] - data["mean_x"] # (n, )
    delta_y = batch_data["mean_y"] - data["mean_y"] # (m, )
    tot_count = data["count"] + batch_data["count"]

    new_mean_x = data["mean_x"] + delta_x * batch_data["count"] / tot_count # (n, )
    new_mean_y = data["mean_y"] + delta_y * batch_data["count"] / tot_count # (m, )

    new_var_x = update_var_from_moments(data["var_x"], data["count"], batch_data["var_x"], batch_data["count"], delta_x, tot_count)
    new_var_y = update_var_from_moments(data["var_y"], data["count"], batch_data["var_y"], batch_data["count"], delta_y, tot_count)

    new_cov = update_cov_from_moments(data["cov"], data["count"], batch_data["cov"], batch_data["count"], delta_x, delta_y, tot_count)

    new_data = {
        "mean_x": new_mean_x,
        "mean_y": new_mean_y,
        "var_x": new_var_x,
        "var_y": new_var_y,
        "cov": new_cov,
        "count": tot_count,
    }

    return new_data


##############################################################################
### USAGE

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
# for _ in range(1000):
#     a = np.random.multivariate_normal(a_mean, a_cov, size=100)
#     running_var_cov.update(a[:, :split_a], a[:, split_a:])

# a_corr = get_corr_matrix(a_cov)
# print(a_corr[:split_a, split_a:]) # for correlation between split_a and split_b

# print(running_var_cov.get_corr_matrix()) # should be close to the output above