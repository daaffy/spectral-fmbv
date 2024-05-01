from node import Node
import numpy as np
from point import Point

class QTree(Node):

    def __init__(self, img_array):
        '''
            QTree
        Quadtree image partition.

        For 2D, for now...

        TO-DO:
        - Set parameters using __init__
        - Convert QTree into an affinity matrix. Get Laplacian from this.
        - How do we do this so that the embedded geometry is presevered as best as possible. Study structure of random walks on coarse graphs.
        '''

        super().__init__()

        self.load_array(img_array)
        self.depth = 0

        self.check, self.nodes, self.points = self._iter()

        self.cr = len(self.nodes)/len(self.points)


    def load_array(self, img_array):
        # Checks...
        # numpy array.

        self.img_array = img_array
        self.nx, self.ny = self.img_array.shape

        # get points
        x = np.linspace(0, self.nx-1, self.nx, dtype='int')
        y = np.linspace(0, self.ny-1, self.ny, dtype='int')
        self.xv, self.yv = np.meshgrid(x, y)

        self.N = len(self.xv.flatten())

        vals = self.img_array.transpose().flatten() # transpose necessary here! check what meshgrid does vs. img_array!!!

        # better way to do this...
        tmp = np.zeros((2,self.N), dtype='int')
        tmp[0,:] = self.xv.flatten()
        tmp[1,:] = self.yv.flatten()
        points_list = []
        for i in range(tmp.shape[1]):
            p = Point((tmp[0,i]+0.5,tmp[1,i]+0.5),vals[i])
            points_list.append(p)

        self.set_points(points_list)
        # self.points = np.zeros((2,self.N))

        self.left = 0
        self.right = self.nx
        self.bottom = 0
        self.top = self.ny # flipped from imshow
        

        # self.points = np.stack((self.xv.flatten(), self.yv.flatten()))

        # self.inc_map = ['' for _ in range(self.N)] # initialise blank inclusion map (all belong to mother node)

        # self.node_mask = np.ones(img_array.shape, dtype='int')

    def get_qimage(self):
        final = np.zeros(self.img_array.shape)
        for p in self.points:
            final[int(p.coords[0]-.5), int(p.coords[1]-.5)] = p.node_val

        return final