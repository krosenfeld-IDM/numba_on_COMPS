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
    n = nb.uint32(1_000_000)

    print("parallel=True")
    f_parallel(5)
    for i in tqdm(range(niter)):
        f_parallel(n)

    print("parallel=False")
    f_no_parallel(5)
    for i in tqdm(range(niter)):
        f_no_parallel(n)

    print(f"Number of threads: {nb.get_num_threads()}")