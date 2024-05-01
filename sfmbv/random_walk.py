import numpy as np
import random
import methods

class RandomWalk():

    """
        TO-DO:
    - Handle sparse matrix. W sparse -> T sparse -> ...
    """

    def __init__(self, G):
        self.G = G
        self.W = methods.construct_adjacency(G, dense = True) # use dense arrays for now (not sparse)
        self.__trans()

    def __trans(self):
        self.T = self.W / np.sum(self.W, axis = 0)

    def _step(self, ind):
        """ _step
        Step forward from a given index. Probabilities are given by T.
        """
        prob = self.T[:,ind]
        r = random.random()
        
        c = 0
        for i in range(len(prob)):
            c = c + prob[i]
            if r <= c:
                return i
    
    def _walk(self, start_ind, length=1):
        """ _walk
        A _walk is a succession of steps of a certain length.
        """
        for i in range(length):
            start_ind = self._step(start_ind)
        return start_ind
    
    def _setup_inc_trans(self, start_ind, inc = None):
        if inc is None:
            sh = self.W.shape
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