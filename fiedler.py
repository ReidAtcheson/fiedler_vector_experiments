import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#Random sparse matrix with nonzero pattern pulled from a normal distribution centered at each row
#Just some reused code to generate sparse matrix. the nonzeros later get discarded because
#we're only interested in graph laplacians here.
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
    #Symmetrize A since we're mainly working with graph laplacians
    A=sp.coo_array((vals,(rids,cids)),shape=(m,m))
    A=A+A.T
    return A

#Quick way to compute graph laplacian
def graph_laplacian(A):
    assert(spla.norm(A-A.T)<1e-14)
    A=A.tocoo()
    G=sp.coo_matrix((-np.ones(A.nnz),(A.row,A.col)))
    G=G-sp.diags([G.diagonal()],[0])
    G = G + sp.diags([-G @ np.ones(G.shape[0])],[0])
    return G

#Simple preconditioner based on excluding nonzeros 
#outside of a specified bandwidth, resulting in 
#very efficient factorizations.
def banded_preconditioner(A,band=5):
    A=A.tocoo()
    rids=[]
    cids=[]
    vals=[]
    for r,c,v in zip(A.row,A.col,A.data):
        if abs(r-c)>band:
            continue
        rids.append(r)
        cids.append(c)
        vals.append(v)
    return sp.coo_matrix((vals,(rids,cids))).tocsc()




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




#Sharper spectral radius bound.
#
#Similar approach to "exact fiedler" above but uses a first lanczos iteration
#to compute a spectral radius that should be sharper estimate than the norm of
#the matrix G, and improve conditioning of the shifted system's large eigenvalues.
def shifted_lanczos(A):
    G = graph_laplacian(A)
    processed_nonzeros=0
    def evalG(x):
        nonlocal processed_nonzeros
        processed_nonzeros += G.nnz
        return G@x
    eigG,_ = spla.eigsh(spla.LinearOperator(G.shape,matvec=evalG), k=6,tol=1e-3)
    specrad = np.max(np.abs(eigG))
    def eval_shiftedG(x):
        nonlocal processed_nonzeros
        processed_nonzeros += G.nnz
        return G@x - specrad*x
    eigG,V = spla.eigsh(spla.LinearOperator(G.shape,matvec=eval_shiftedG), k=6, tol=1e-6)
    #The smallest eigenvalue should be \approx 0
    #assert(abs(eigG[0]+specrad)<1e-10)
    return eigG[1]+specrad,V[:,1],processed_nonzeros

#Unpreconditioned LOBPCG
#
def lobpcg_noprecon(A,k=6,rng=None,tol=1e-2):
    if rng is None:
        rng = np.random.default_rng(0)
    G = graph_laplacian(A)
    processed_nonzeros=0
    def evalG(x):
        nonlocal processed_nonzeros
        processed_nonzeros += G.nnz
        return G@x

    X = rng.uniform(-1,1,size=(G.shape[0],k))
    #Go ahead and put the nullspace in `X` the algorithm
    #should register it as converged immediately and project it
    #out of the remaining eigenvectors
    X[:,0] = np.ones(G.shape[0])
    eigG,V = spla.lobpcg(spla.LinearOperator(G.shape,matvec=evalG),X,largest=False,maxiter=200,tol=tol)
    ids=np.argsort(eigG)
    eigG=eigG[ids]
    V = V[:,ids]
    return eigG[1],V[:,1],processed_nonzeros



#Unpreconditioned LOBPCG
#
def lobpcg_banded_precon(A,k=6,tol=1e-2,rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    G = graph_laplacian(A)
    processed_nonzeros=0
    band=5
    Gh = banded_preconditioner(G,band=band)
    #Note: since we have excluded some off-diagonal entries the 
    #resulting matrix is usually invertible.
    luGh = spla.splu(Gh)
    def evalGh(x):
        nonlocal processed_nonzeros
        #This is an approximate number of nonzeros 
        #that the total factorization will contain.
        #Technically if sparse LU does any pivoting then the bandwidth can increase
        #to 2*band (resulting in 4*band*G.shape[0] nonzeros), but it's unlikely
        #here because G is semidefinite (and Gh should become _definite_ after
        #discarding some off-diagonal entries)
        processed_nonzeros += 2*band*G.shape[0]
        return luGh.solve(x)
    def evalG(x):
        nonlocal processed_nonzeros
        processed_nonzeros += G.nnz
        return G@x

    X = rng.uniform(-1,1,size=(G.shape[0],k))
    #Go ahead and put the nullspace in `X` the algorithm
    #should register it as converged immediately and project it
    #out of the remaining eigenvectors
    X[:,0] = np.ones(G.shape[0])
    eigG,V = spla.lobpcg(spla.LinearOperator(G.shape,matvec=evalG),X,M = spla.LinearOperator(G.shape,matvec=evalGh),largest=False,maxiter=200,tol=tol)
    ids=np.argsort(eigG)
    eigG=eigG[ids]
    V = V[:,ids]
    return eigG[1],V[:,1],processed_nonzeros






























