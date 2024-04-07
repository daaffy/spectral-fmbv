import numpy as np

class Node:

    def __init__(self, label='', depth=None, bounds=[None,None,None,None]):
        '''
            Node
        Node object for quadtree partition.

        TO-DO:
        - Need a robust set_points function

        '''

        # print("depth: "+str(depth))

        self.depth = depth
        self.label = label
        self.is_leaf = True # a fresh node is a leaf

        self.points = []
        self.n_points = 0

        self.node_mask = None # idk if we want to do it like this...

        self.daughters = []

        # daughters
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.left = bounds[0]
        self.right = bounds[1]
        self.bottom = bounds[2]
        self.top = bounds[3]
        
        # change this from hard-coded...
        self.max_depth = 6
        self.std_thresh = 4

    def partition(self):
        '''
            partition
        Operation divides current node into four daughter nodes with spatial quadrant organisation.
        '''
        if not self.is_leaf:
            raise Exception("Node has already been partitioned...")

        # cycle through points and sort into quadrants
        a_points, b_points, c_points, d_points = [], [], [], []

        # define the centres of the quadrant 
        cx = int(self.left + .5*(self.right - self.left))
        cy = int(self.bottom + .5*(self.top - self.bottom))

        for p in self.points:
            x = p.coords[0]
            y = p.coords[1]

            if x < cx:
                if y < cy:
                    # c
                    c_points.append(p)
                else:
                    # a
                    a_points.append(p)
            else: # x >= cx
                if y < cy:
                    # d
                    d_points.append(p)
                else:
                    # b
                    b_points.append(p)
            
        # 'a', north-west
        self.a = Node(label=self.label+'a', depth=self.depth+1, 
                      bounds=[self.left,cx,cy,self.top]) # (!) add depth...
        # self.a.points = self.points[0]
        self.a.set_points(a_points) # test: progressively remove one point, leave the other nodes empty.

        # 'b', north-east
        self.b = Node(label=self.label+'b', depth=self.depth+1, 
                      bounds=[cx,self.right,cy,self.top])
        self.b.set_points(b_points) # test: progressively remove one point, leave the other nodes empty.

        # 'c', south-west
        self.c = Node(label=self.label+'c', depth=self.depth+1, 
                      bounds=[self.left,cx,self.bottom,cy])
        self.c.set_points(c_points)

        #'d', south-east
        self.d = Node(label=self.label+'d', depth=self.depth+1, 
                      bounds=[cx,self.right,self.bottom,cy])
        self.d.set_points(d_points)

        # ...
        self.daughters = [self.a, self.b, self.c, self.d]
        self.is_leaf = False # current node is no longer a leaf after it has been partitioned...

    def _iter(self):
        '''
            _iter
        This is where we iterate over the image and form the quad-tree structure.

        '''

        if not self.is_leaf:
            raise Exception("_iter can only be run on a leaf node...")

        avg = 0
        val_list = []
        if self.n_points > 0:
            for p_ in self.points:
                avg = avg + p_.val # could use val_list for this.
                val_list.append(p_.val)
            avg = avg / self.n_points
        std = np.std(val_list)

        if self.n_points <= 1 or self.depth >= self.max_depth or std < self.std_thresh:
            # if: homogeneous pixel values or only one pixel left or if depth too deep...
            #   do nothing, finalise...
            
            # --- calculate average
            # # print('im a leaf and i\'m '+self.label)
            # if self.n_points == 1:
            #     print(self.points[0].coords)
            # print([self.points[i].node_id for i in range(len(self.points))])
            
            # print(avg)

            for p_ in self.points:
                p_.assign_node(self.label, avg)
            return True, [self], self.points #, return inclusion map
        else:
            # else: (can still dig deeper...)
            #   partition...
            #   for over daughters[i]._iter()...
            self.partition()

            check = True # check that all iterations have succeeded.
            node_list = []
            points_list = []
            for i in range(4): # cycle over daughter nodes.
                tmp_check, tmp_nodes, tmp_points = self.daughters[i]._iter()
                check = check*tmp_check
                node_list.extend(tmp_nodes)
                points_list.extend(tmp_points)

            return check, node_list, points_list

    def set_points(self, points):
        self.points = points
        self.n_points = len(points)

    # def _fetch(self):
    #     if self.is_leaf == True:
    #         return self.points
    #     else:
    #         for node in self.daughters:
        
    # def 



        

        