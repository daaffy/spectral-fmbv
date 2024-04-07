import scipy
import numpy as np

class Graph():

    def __init__(self, verbose = False):
        '''
            Graph
        Object to handle graph storage and identification procedures.
        '''

        self.nodes = None   # node labels (necessary?)
        self.adj = None     # adjacency matrix; connectivity information.
        self.inc = None     # inclusion map; begins as the nodes identity map.

        self.verbose = verbose

    def set(self, nodes, adj):
        # check size match...
        (n1, n2) = adj.shape

        assert n1 == n2
        assert n1 == len(nodes)
        # assert sparsity
        # assert adj bool datatype

        # set nodes and adjacency matrix
        self.nodes = nodes
        self.adj = adj
        self.inc = nodes

        # store original
        self.nodes0 = nodes
        self.adj0 = adj

        self._vrbs("Adjacency matrix: ")
        self._vrbs(self.adj)
    
    def identify(self, ids):
        # ids must be a list of tuples.
        self._elim(ids[0])

    def _elim(self, id):
        self._vrbs("Eliminating... ")

        assert isinstance(id, tuple)
        self._vrbs(id)

        hold = id[0]
        kill = id[1]

        temp = self.adj[kill,:]
        # temp[0,kill] = False # ignore connection between hold and kill nodes
        self._vrbs(temp.toarray())

        temp = scipy.sparse.coo_matrix(temp)
        temp_row, temp_col, temp_data = [], [], []
        for i,j,v in zip(temp.row, temp.col, temp.data):
            temp_row.append(i)
            temp_col.append(j)
            if i == j:
                temp_data.append(0)
            else:
                temp_data.append(v)

            if not i == j:
                temp_row.append(j)
                temp_col.append(i)
                temp_data.append(v)
        temp = scipy.sparse.coo_matrix((temp_data, (temp_row, temp_col)), dtype=bool)
        temp = scipy.sparse.csr_matrix(temp)
        # self._vrbs(temp.toarray())

        self.adj = self.adj + temp

        # delete kill row and column
        # self.adj[kill,:]
            

    def _vrbs(self, obj):
        # basic debugging tool
        if self.verbose:
            print(obj)

        

class RW_Graph():

    def __init__(self, W):
        self.W = W # as array.
        self._trans()
    
    def _trans(self):
        self.T = self.W / np.sum(self.W, axis = 0)

    def _step(self, ind):
        prob = self.T[:,ind]
        r = random.random()
        
        c = 0
        for i in range(len(prob)):
            c = c + prob[i]
            if r <= c:
                return i
            
    def _walk(self, start_ind, length=1):
        for i in range(length):
            start_ind = self._step(start_ind)
        return start_ind
    
    def _setup_inc_trans(self, start_ind, inc = None):
        if inc.any() == None:
            sh = W.shape
            self.inc = np.array(range(sh[0]))
        else:
            # checks...
            self.inc = inc
        
        self.state_to_ind = []
        self.uniq = []
        c = -1
        for i in self.inc:
            if not i in self.uniq:
                self.uniq.append(i)
                c = c + 1
            self.state_to_ind.append(c)

        inc_n = len(self.uniq)
        self.inc_T = np.zeros((inc_n, inc_n), dtype=int)

        self.start_ind = start_ind

    def _step_inc_trans(self):
        a = self.start_ind
        self.start_ind = self._step(self.start_ind)
        b = self.start_ind

        inc_a = self.state_to_ind[a]
        inc_b = self.state_to_ind[b]
        # print(inc_b)
        self.inc_T[inc_b,inc_a] = self.inc_T[inc_b,inc_a] + 1
