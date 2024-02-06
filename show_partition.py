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

mx=32
my=32
m = mx*my
A = fiedler.random_uniform_sparse(m,7,32,rng=rng)
e,f,_ = fiedler.exact_fiedler(A)
ids = np.argsort(f)

#The matrix A is random and has no inherent geometric interpretation
#however because the nonzero pattern is generated with a normal distribution
#this gives A a naturally bounded bandwidth which means that _if we were_ to
#interpret its nodes as gridpoints the node connectivity would be fairly local
#thus the optimal partition ends up being just a simple split midway through
#the grid
im = np.zeros(mx*my)
im[ids[0:m//2]]=1.0
plt.imshow(im.reshape((mx,my)))
plt.savefig("part.svg")
