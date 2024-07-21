from numba import njit, prange
import numpy as np
from tqdm import tqdm

@njit(parallel=True)
def two_d_array_reduction_prod(n):
    shp = (13, 17)
    result1 = 2 * np.ones(shp, np.int_)
    tmp = 2 * np.ones_like(result1)

    for i in prange(n):
        result1 *= tmp

    return result1

if __name__ == "__main__":
    niter = 100
    n = 10_000
    for i in tqdm(range(niter)):
        two_d_array_reduction_prod(n)