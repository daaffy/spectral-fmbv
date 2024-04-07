"""

Jack J. Hills

diffusion maps etc. that act on networkx graph objects.
random walks.

preset visualisation

"""

import numpy as np
from scipy import spatial
import networkx as nx
import SimpleITK as sitk
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy #.sparse ?

class WeightKernel():
    def __init__(
            self,
            dist_sig = 1,
            intensity_sig = 1
            ):
        self.dist_sig = dist_sig
        self.intensity_sig = intensity_sig
    
    def eval(self, pixel_dist, intensity_dist):
        tmp = np.exp(-pixel_dist**2/ self.dist_sig**2) * np.exp(-intensity_dist**2/ self.intensity_sig**2)
        return tmp

# class ImageGraphTest(nx.Graph):
#     def __init__(
#             self
#     ):
#         super().__init__()

def construct_adjacency(G, dense = False):
    """ construct_adjacency
    Build an adjacency array from the graph data in G.
    """

    nodes_list = list(G.nodes)

    row = []
    col = []
    dat = []

    # note this is an inefficient way for now...
    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            n1, n2 = nodes_list[i], nodes_list[j]
            temp = G.get_edge_data(n1,n2) # n1, n2 order convention?
            w = 0
            if not temp is None:
                w = temp['weight']
            
            row.append(i)
            col.append(j)
            dat.append(w)

    sparse_adj = scipy.sparse.coo_matrix((dat, (row, col)))

    if dense:
         return sparse_adj.toarray()
    else:
         return sparse_adj

def stationary_distribution(G):
    dist = {}
    dist_list = []
    total = 0
    for node in G.nodes:
        total = total + G.degree(node)

    for node in G.nodes:
        temp = G.degree(node)/total
        dist[node] = temp
        dist_list.append(temp)

    return dist, np.array(dist_list)

class CoarseGraph(nx.Graph):

    def __init__(self, init_G, partition, mode = "coifman_06"):
        self.init_G = init_G
        self.partition = partition

        self.partition_ids = list(np.unique(partition))

        super().__init__()

        if mode == "coifman_06":
            self.__construct_coarse_lafon_06()
        else:
            raise Exception("Mode not recognised.")

        # checks:
        # is the partition formatted correctly?
        # ...

    def __construct_coarse_lafon_06(self):
        """
        Lafon and Lee. Diffusion Maps and Coarse-Graining:
        A Unified Framework for Dimensionality
        Reduction, Graph Partitioning, and
        Data Set Parameterization. (Section 3.1).
        """

        st, _ = stationary_distribution(self.init_G)

        def __add_weights(nodes1, nodes2):
            w_prime = 0
            for n1 in nodes1:
                for n2 in nodes2:
                    temp = self.init_G.get_edge_data(n1,n2)
                    if temp is None:
                        w = 0
                    else:
                        w = self.init_G.get_edge_data(n1,n2)['weight'] * st[n1] # need stationary dist; st[n1 or n2?]
                    w_prime = w_prime + w
            return w_prime
    
        for x in self.partition_ids:
            for y in self.partition_ids:
                self.add_edge(x,y,weight=__add_weights(
                     self.__get_nodes(x),
                     self.__get_nodes(y)
                ))
        0
    
    def __get_nodes(self, partition_id):
        return list(np.array(self.init_G.nodes)[np.array(self.partition) == partition_id])

class ImageGraph(nx.Graph):
    """
        ImageGraph object handles the creation of NetworkX graph objects from multi-dimensional array/image data. 

        Class structure allows for easy masking and reinterpreting graph eigenvectors as images, for example.
    """

    def __init__(
            self,
            img_array,
            mask_array = None,
            wf = WeightKernel(), # Weighting Function (Kernel)
            neighbour_depth = 1,
            cmap = None
            ):
        
        super().__init__()
        
        self.sh = img_array.shape
        self.img_array = img_array
        self.wf = wf
        self.neighbour_depth = neighbour_depth
        self.mask_array = mask_array
        self.cmap = cmap

        self.dist, self.v_coords = _distance_information_from_img_array(img_array, neighbour_depth=neighbour_depth)
        
        self._img_array_to_graph()
        # self.G = _img_array_to_graph(img_array, mask_array = self.mask_array, neighbour_depth=self.neighbour_depth, pixel_sig=5, img_sig=30)

    def __set_node_colour(self):
        """
            __set_node_colour
        Set the nodes to colour values defined by 'cmap' and based on image intensity values.

        TO-DO:
        - digitizing the whole array might be inefficient from a storage perspective; can manually bin each pixel intensity as we go...
        """
        bins = np.linspace(0, 255, 256, dtype=int)
        scale_v = np.digitize(self.img_array, bins) / 255 
        
        map = plt.get_cmap(self.cmap)

        for i in list(self.nodes): # i is not a slice but the node index itself
            curr_v = scale_v[*self.v_coords[i]]
            r, g, b, a = map(curr_v)

            self.nodes[i]['viz'] = {'color': {'r': np.int(255*r), 'g': np.int(255*g), 'b': np.int(255*b), 'a': np.int(1*a)}} # a <-> 1?

    def _img_array_to_graph(
            self,
            # wf = WeightKernel()
            # dist_sig = 1,
            # intensity_sig = 1
            ):
        
        if self.mask_array is None:
            mask_array = np.full(self.sh, True)
        else:
            mask_array = self.mask_array

        # N = self.dist.shape[0]

        for i in range(self.dist.nnz): # loop over all non-zero elements of the distance array
            if self.dist.data[i] == 0:
                continue

            j1, j2 = self.dist.row[i],  self.dist.col[i]

            # check if either of the edge nodes belong to mask.
            if not mask_array[*self.v_coords[j1,:]] or not mask_array[*self.v_coords[j2,:]]:
                # ...
                continue
            
            # then calculate the intensity difference
            intensity_dist = np.abs(self.img_array[*self.v_coords[j1,:]]-self.img_array[*self.v_coords[j2,:]])

            # weight (affinity) between nodes depends on distance in both pixel distance and pixel intensity space.
            # w = _w1(self.dist.data[i], intensity_dist,  dist_sig=dist_sig, intensity_sig=intensity_sig)
            w = self.wf.eval(self.dist.data[i], intensity_dist)
            
            # # clean up some of the small small numbers.
            # if w < 1.e-4: # might not keep this.
            #     w = 0
            
            self.add_edge(j1,j2, weight=w)
        
        if not self.cmap is None:
            self.__set_node_colour()

    # def _calculate_eigen(self):
    #     self.N = nx.normalized_laplacian_matrix(self)
    #     self.vals, self.vecs = scipy.sparse.linalg.eigs(self.N,which='SR')

    def _node_vec_to_img_array(self, in_vec):
        # check vector is the right size
        assert len(in_vec) == len(self.nodes)

        out_vec = np.zeros(np.prod(self.sh))
        for i in range(len(self.nodes)):
            out_vec[list(self.nodes)[i]] = in_vec[i]

        return np.transpose(out_vec.reshape(self.sh[1], self.sh[0])) # not sure about why transpose works.
        # return np.transpose(out_vec.reshape(*self.sh)) # not sure about why transpose works.


# class ImageGraph(nx.Graph):
#     """
#         ImageGraph object handles the creation of NetworkX graph objects from
#         multi-dimensional array/image data. 

#         Class structure allows for easy masking and reinterpreting graph eigenvectors as images, for example.
#     """

#     def __init__(
#             self,
#             img_array,
#             mask_array = None,
#             neighbour_depth = 1
#             ):
        
#         self.img_array = img_array
#         self.neighbour_depth = neighbour_depth
#         self.mask_array = mask_array

#         self.G = _img_array_to_graph(img_array, mask_array = self.mask_array, neighbour_depth=self.neighbour_depth)
        
def _w0(pixel_dist, intensity_dist, dist_sig = 1, intensity_sig = 1):
    """
        _w0
    pixel_sig: related to physical distance between pixels.
    img_sig  : related to distance between pixel intensities.

    * this is a bit ambiguous, needs changing. 
    """
    tmp = np.exp(-pixel_dist**2/ dist_sig**2) * np.exp(-intensity_dist**2/ intensity_sig**2)
    return tmp

def _w1(pixel_dist, *vargs, **kwargs):
    return pixel_dist

def _distance_information_from_img_array(img_array, neighbour_depth):
    EUC_DISTANCE_P = 2

    sh = img_array.shape
    dims = len(sh)

    # Use meshgrid to find attached coordinates to image intensities.
    tmp = np.meshgrid(*[range(sh[i]) for i in range(dims)]) # create coordinate arrays.
    v_coords = np.concatenate(
        tuple([tmp[j].flatten()[:, None] for j in range(dims)]), 
        axis=1) # flatten into a vertical array; each row is a point in image space.

    # Use a kd-tree to construct a sparse adjacency matrix.
    kd_tree = spatial.KDTree(v_coords)
    dist = kd_tree.sparse_distance_matrix(kd_tree, neighbour_depth, p=EUC_DISTANCE_P, output_type='coo_matrix')

    return dist, v_coords



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



def _img_array_to_graph(
        img_array,
        mask_array = None,
        neighbour_depth = 1,
        pixel_sig = 1,
        img_sig = 1
        ):
    """ Constructs a NetworkX graph object from an image array.
        
    """

    # checks... e.g., mask_array dimensions are the same as img_array.
    # mask array is bool data.

    if mask_array is None:
        mask_array = np.full(img_array.shape, True)

    dist, v_coords = _distance_information_from_img_array(img_array, neighbour_depth=neighbour_depth)
    N = dist.shape[0]
    # Create graph object.
    G = nx.Graph()

    # mask_inds = mask_array.flatten()

    # G.add_nodes_from(list(range(N))) # add nodes immediately so that laplacian is ordered properly; not good when masking.

    # iterate over possible edges
    for i in range(dist.nnz):
        if not dist.data[i] == 0:
            j1, j2 = dist.row[i],  dist.col[i]

            # check if either of the edge nodes belong to mask.
            if not mask_array[*v_coords[j1,:]] or not mask_array[*v_coords[j2,:]]:
                # ...
                continue

            img_dist = np.abs(img_array[*v_coords[j1,:]]-img_array[*v_coords[j2,:]])

            # weight (affinity) between nodes depends on distance in both pixel distance and pixel intensity space.
            w = _w0(dist.data[i], img_dist,  pixel_sig=pixel_sig, img_sig=img_sig)
            
            # clean up some of the small small numbers.
            if w < 1.e-4: # might not keep this.
                w = 0
            
            G.add_edge(j1,j2, weight=w)
   
    return G


# Preset 2-dimensional image arrays.

def dots_2d(
        noise = 20
        ):
    """ Returns a 2-dimensional array of dots with noise
    
    """
    n = 50

    noise_mag = noise

    img_array = np.zeros((n,n))

    def dcirc(xy, r, mag):
        x, y = xy[0], xy[1]

        temp = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if (i-x)**2 + (j-y)**2 <= r**2: # inside the circle
                    temp[i,j] = mag

        return temp

    img_array = img_array + dcirc([20,20], 7, 255)
    img_array = img_array + dcirc([50,40], 20, 200)
    img_array = img_array + dcirc([10,40], 5, 220)
    img_array = img_array + dcirc([0,0], 10, 150)
    img_array = img_array + dcirc([40,10], 6, 100)

    img_array = img_array + noise_mag*np.random.random((n,n))

    return img_array, img_array.shape

def blank(n = 10, noise=20):
    img_array = noise*np.random.random((n,n))
    return img_array, img_array.shape

def pd(zoom = 1):
    
    base = 'wl1_5'
    # path = '/Users/jackh/Documents/FMBV_2023/gordon_original/experiments/ParameterStudy/data_prelim_01/IMG_20240212_3_1_dp.nii.gz'
    path = '/Users/jackh/Documents/FMBV_2023/gordon_original/test_batch/'+base+'_dp.nii.gz'

    img = sitk.ReadImage(path)
    pd_array = sitk.GetArrayFromImage(img).astype('float64')

    img_array = pd_array[:,int(.5*pd_array.shape[1])-0,:]

    # resample
    img_array = scipy.ndimage.zoom(img_array, zoom, order=1)

    return img_array, img_array.shape
