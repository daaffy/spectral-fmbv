
import numpy as np

"""
SEBA
Python implementation of the SEBA algorithm.
Adapted from Julia code written by Gary Froyland.
"""
def SEBA(V, Rinit = None):
    # V is pxr matrix (r vectors of length p as columns)
    # Rinit is an (optional) initial rotation matrix.

    # Outputs:
    # S is pxr matrix with columns approximately spanning the column space of V
    # R is the optimal rotation that acts on V, which followed by thresholding, produces S

    # Begin SEBA algorithm
    maxiter = 5000   # maximum number of iterations allowed
    F,_ = np.linalg.qr(V) # Enforce orthonormality
    V = F # (!) needed?
    (p,r) = np.shape(V)
    mu = 0.99 / np.sqrt(p)

    S = np.zeros(np.shape(V))

    # Perturb near-constant vectors
    for j in range(r):
            if np.max(V[:, j]) - np.min(V[:, j]) < 1e-14:
                    V[:, j] = V[:, j] + (np.random.random((p, 1)) - 1 / 2) * 1e-12

    # is R correct?

    # ...
    # Initialise rotation
    if Rinit == None:
            Rnew = np.eye(r) # depends on context?
    else:
            # Ensure orthonormality of Rinit
            U, _, Vt = np.linalg.svd(Rinit)
            Rnew = np.matmul(U , Vt)

    #preallocate matrices
    R = np.zeros((r, r))
    Z = np.zeros((p, r))
    Si = np.zeros((p, 1))

    iter = 0
    while np.linalg.norm(Rnew - R) > 1e-14 and iter < maxiter:
            iter = iter + 1
            R = Rnew
            Z = np.matmul(V , R.T)

            # Threshold to solve sparse approximation problem
            for i in range(r):
                    Si = soft_threshold(Z[:,i], mu)
                    S[:, i] = Si / np.linalg.norm(Si)
            # Polar decomposition to solve Procrustes problem
            U, _, Vt = np.linalg.svd(np.matmul(S.T , V), full_matrices=False)
            Rnew = np.matmul(U , Vt)

    # Choose correct parity of vectors and scale so largest value is 1
    for i in range(r):
            S[:, i] = S[:, i] * np.sign(sum(S[:, i]))
            S[:, i] = S[:, i] / np.max(S[:, i])

    # Sort so that most reliable vectors appear first
    ind = np.argsort(np.min(S, axis=0))
    S = S[:, ind]

    return S, R

def soft_threshold(z, mu):
    assert len(np.shape(z)) <= 1 # only accept scalars or vectors

    temp = np.zeros(np.shape(z))
    if len(np.shape(z)) == 1:
            for i in range(len(z)):
                    temp[i] = np.sign(z[i]) * np.max([np.abs(z[i]) - mu, 0])
    else:
            temp = np.sign(z) * np.max([np.abs(z) - mu, 0])        
    
    return temp


