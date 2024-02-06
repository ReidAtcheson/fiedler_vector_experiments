import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#Random sparse matrix with nonzero pattern pulled from a normal distribution centered at each row
def random_uniform_sparse(m,nnz,std,rng=None):    
    if rng is None:    
        rng=np.random.default_rng(0)    
    rids=[]    
    cids=[]    
    vals=[]    
    for i in range(m):    
        cs=[int(rng.normal(loc=i,scale=std)) for _ in range(nnz)] + [i]    
        cs=[min(max(0,ci),m-1) for ci in cs]    
        cs=set(cs)    
        for ci in cs:    
            rids.append(i)    
            cids.append(ci)    
            vals.append(rng.uniform(-1,1))    
    return sp.coo_array((vals,(rids,cids)),shape=(m,m))


#Quick way to compute graph laplacian
def graph_laplacian(A):
    #Symmetrize A
    A=A+A.T
    A=A.tocoo()
    G=sp.coo_matrix((-np.ones(A.nnz),(A.row,A.col)))
    G=G-sp.diags([G.diagonal()],[0])
    G = G + sp.diags([-G @ np.ones(G.shape[0])],[0])
    return G


#"Exact" fiedler vector/eigenvalue pair.
#
#Since we are working with large and sparse matrices it's not possible to have a full guarantee
#that we have computed the second-smallest eigenvalue, but we can require a very stringent
#convergence tolerance alongside a large window of Krylov vectors and because of the way
#Krylov methods work it is unlikely that the converged result omits the fiedler value.
def exact_fiedler(A):
    G = graph_laplacian(A)
    normG = spla.norm(G)
    #Replacing G with G - ||G||*I and using an Arnoldi method
    #effectively turns the problem of finding the smallest eigenvalues
    #of G to that of finding the largest (in magnitude) eigenvalues of G - ||G||*I.
    #This way we do not need to solve shifted linear systems.
    processed_nonzeros=0
    def evalG(x):
        nonlocal processed_nonzeros
        processed_nonzeros += G.nnz
        return G@x - normG*x
    eigG,V = spla.eigsh(spla.LinearOperator(G.shape,matvec=evalG), k=100)
    #The smallest eigenvalue should be \approx 0
    assert(abs(eigG[0]+normG)<1e-10)
    return eigG[1]+normG,V[:,1],processed_nonzeros















