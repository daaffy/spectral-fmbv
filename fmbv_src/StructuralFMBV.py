'''
    An extension of FMBV; graph-theoretic methods of standardising Power Doppler data that
    attempt to take vessel structure into account.
'''

from fmbv_refactor import FMBV
import scipy
import numpy as np

class StructuralFMBV(FMBV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.L = None

    #   ---------------------------------------
    #   GRAPH LAPLACIAN METHODS
    #   ---------------------------------------

    '''
        w0
    Basic Gaussian kernel weighting.
    '''
    def w0(self,
           D,
           v_ind1, 
           v_ind2,
           sig=1,
           ind_sig=1):
        

        ind_d = np.linalg.norm(
                np.array([v_ind1[0],v_ind1[1],v_ind1[2]]) - np.array([v_ind2[0],v_ind2[1],v_ind2[2]])
        )
        d1 = D[v_ind1[0],v_ind1[1],v_ind1[2]]
        d2 = D[v_ind2[0],v_ind2[1],v_ind2[2]]
        return np.exp(-(d1-d2)**2/sig**2) * np.exp(-ind_d**2/ind_sig**2), ind_d

    '''
        _get_laplacian
    Calculate normalised (optional) graph Laplacian on PD image data.
    '''
    def _get_laplacian(self, 
                       eps=1, 
                       sig=1,
                       ind_sig=1,
                       threshold = 0.001,
                       normalised=False,
                       keep=None,
                       mode='default'):
        
        if not self.pd_supplied:
            raise Exception("PD data not supplied.")

        if not len(self.pd_array.shape) == 3:
            raise Exception("PD array must be 3-dimensional.")

        self.n_x, self.n_y, self.n_z = self.pd_array.shape[0], self.pd_array.shape[1], self.pd_array.shape[2]
        

        # Use KD-Tree to construct adjacency matrix
        X, Y, Z = np.meshgrid(range(self.n_x),range(self.n_y),range(self.n_z))
        v_coords = np.concatenate((X.flatten()[:,None],Y.flatten()[:,None],Z.flatten()[:,None]), axis=1)

        self.n = v_coords.shape[0] # n (pre-masked)

        if not keep is None: # if we are masking out some of the values...
            keep_vec = keep.flatten()
            self.map_back = np.array(range(self.n))[keep_vec]
            v_coords = v_coords[keep_vec]

        # print(map_back)
            
        self.vrbs("kd_tree...")

        kd_tree = scipy.spatial.KDTree(v_coords)
        dist = kd_tree.sparse_distance_matrix(kd_tree, eps, p=2, output_type='coo_matrix')

        nnz = dist.nnz
        
        self.vrbs("constructing laplacian...")

        W = dist
        for i in range(W.nnz):
            if mode == 'hard':
                # hard: basic adjacency matrix, only directly adjacent nodes are connected and unweighted by distance
                w, ind_d = self.w0(self.pd_array,v_coords[W.row[i]],v_coords[W.col[i]],sig=sig,ind_sig=np.inf)
                if ind_d <= ind_sig and ind_d > 0:
                # and ind_d > 0:
                    W.data[i] = 1
                else:
                    W.data[i] = 0
            else:
                W.data[i], ind_d = self.w0(self.pd_array,v_coords[W.row[i]],v_coords[W.col[i]],sig=sig,ind_sig=ind_sig)
                if ind_d == 0: # could be redundant
                    W.data[i] = 0

            if W.data[i] <= threshold:
                W.data[i] = 0

            

            print(str(int(100*i/W.nnz)),end='\r')

        # Degree matrix
        deg = np.sum(W,axis=1) # !!!
        deg = np.squeeze(np.asarray(deg))
        deg = scipy.sparse.diags(deg,0)

        self.L = deg - W

        self.W = W

        return self.L, deg
    
    def _reshape(self, vec):
        pad_vec = np.zeros((self.n))
        pad_vec[self.map_back] = vec
        return np.reshape(pad_vec,(self.n_x, self.n_y, self.n_z)), pad_vec
    
    """
    SEBA
    Python implementation of the SEBA algorithm.
    Adapted from Julia code written by Gary Froyland.
    """
    def SEBA(self, V, Rinit = None):
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
                        Si = self.soft_threshold(Z[:,i], mu)
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

    def soft_threshold(self, z, mu):
                assert len(np.shape(z)) <= 1 # only accept scalars or vectors

                temp = np.zeros(np.shape(z))
                if len(np.shape(z)) == 1:
                        for i in range(len(z)):
                                temp[i] = np.sign(z[i]) * np.max([np.abs(z[i]) - mu, 0])
                else:
                        temp = np.sign(z) * np.max([np.abs(z) - mu, 0])        
                
                return temp
