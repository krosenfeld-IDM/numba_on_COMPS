import numba as nb
from numba import njit, prange
import numpy as np
from tqdm import tqdm

@njit((nb.uint32,), parallel=True, nogil=True, cache=True)
def f_parallel(n):
    shp = (13, 17)
    result1 = 2 * np.ones(shp, np.int_)
    tmp = 2 * np.ones_like(result1)

    for i in prange(n):
        result1 *= tmp

    return result1

@njit((nb.uint32,), parallel=False, nogil=True, cache=True)
def f_no_parallel(n):
    shp = (13, 17)
    result1 = 2 * np.ones(shp, np.int_)
    tmp = 2 * np.ones_like(result1)

    for i in prange(n):
        result1 *= tmp

    return result1

if __name__ == "__main__":

    niter = 100
    n = nb.uint32(100_000)
    f_parallel(5)
    for i in tqdm(range(niter)):
        f_parallel(n)

    f_no_parallel(5)
    for i in tqdm(range(niter)):
        f_no_parallel(n)