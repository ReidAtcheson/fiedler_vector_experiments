import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fiedler

seed=23984723
rng=np.random.default_rng(seed)

m = 10000
A = fiedler.random_uniform_sparse(m,7,32,rng=rng)
fe,_,processed_nonzeros_e = fiedler.exact_fiedler(A)
fh,_,processed_nonzeros_h = fiedler.shifted_lanczos(A)

print(fe,fh,processed_nonzeros_e/A.nnz,processed_nonzeros_h/A.nnz)
