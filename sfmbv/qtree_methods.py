import numpy as np
import copy

"""
TO-DO

    The settings_dict thing is a bit clunky and redundant in some places.
    Ignore parts of the image with mask.

"""

class Node(tuple):

    def __new__ (cls, inds, **kwargs):
        assert isinstance(inds, tuple)
        return super(Node, cls).__new__(cls, inds)

    def __init__(
            self, 
            inds,
            id=[], # id will be a kind of coordinate in the qtree 
            img_array=None,
            depth=None, 
            # bounds=[None,None,None,None], # can we get rid of this? make variable length tuple (for N-dim)?
            min_depth = None,
            max_depth = None, 
            std_thresh = None,
            settings_dict = None
            ):
        
        """
        # Each node is quad-region of the image (indicated by bounds?) that contains a set of Pixel objects.

        Indices are understood to be lower inclusive and upper exclusive (standard slicing conventions). or maybe not. :/
        """

        assert self.__check_tuple(self) # check formatting
        self.dim = len(self)

        self.depth = depth
        self.id = id
        self.is_leaf = True # a fresh node is a leaf

        self.img_array = img_array
        assert len(np.shape(self.img_array)) == len(self) or self.img_array is None

        self.img_avg, self.img_std =  self._calculate_img_statistics()

        # self.pixels_are_set = False
        # self.pixels = []
        # self.n_pixels = 0

        if settings_dict is None:
            self.set_settings()
        else:
            self.settings_dict = settings_dict

        # division
        # self.midpoint = None
        self._calculate_midpoint()
        self.daughter_nodes = []

        # parameters...
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.std_thresh = std_thresh

    def set_settings(self, min_depth = 0, max_depth = 5, std_delta = 1):
        # should this (settings) be a small class instead?
        self.settings_dict = {}
        self.settings_dict['min_depth'] = min_depth
        self.settings_dict['max_depth'] = max_depth
        self.settings_dict['std_delta'] = std_delta

    # DIVISION...

    def _calculate_midpoint(self, *args):
        if len(args) == 0:
            temp = []
            for x in self: # change self to args[0] and use self optionally!
                temp.append(int(.5*(x[0]+x[1])))
            self.midpoint = tuple(temp)
        else: # idk if i like this...
            x = args[0]
            return int(.5*(x[0]+x[1]))

    def _get_list_of_daughter_inds(self, inds, midpoint):
        """ __get_list_of_daughter_inds
        Extract list of daughter indices (recursively)

        * I don't know how to do this in a more elegant way (how do we nest a variable amount of for loops?).
        """
        assert self.__check_tuple(inds)
        assert isinstance(midpoint, tuple)
        assert len(inds) == len(midpoint)

        if inds[0][0] + 1 == inds[0][1]:
            # return 
            curr_div = [[(inds[0][0], inds[0][1])]]
            # curr_div = [[(0,5)]]
            # curr_div = []
        else:
            curr_div = [[(inds[0][0], midpoint[0])], [(midpoint[0], inds[0][1])]]
        
        if len(inds) == 1:
            return curr_div
        else:
            loop_over = self._get_list_of_daughter_inds(inds[1:], midpoint[1:])
            temp = []
            for i in range(len(curr_div)):
                for j in loop_over:
                    # if j is None:
                    #     continue
                    temp.append(curr_div[i]+j)
            return temp

    def _divide(self, iter = False):

        if not self.is_leaf:
            raise Exception("Node already divided.")
        


        if iter and not (self.depth < self.settings_dict['max_depth'] and self.img_std > self.settings_dict['std_delta']):
            if self.depth >= self.settings_dict['min_depth']:
                return 

        temp_list = self._get_list_of_daughter_inds(self, self.midpoint) # beware: list of list of tuples not list of tuples (we must convert before we create Node)
        
        # if temp_list is None:
        #     return

        if len(temp_list) == 1:
            return

        for i in range(len(temp_list)):
            # print(self.id.append(i))
            curr_node = Node(
                    tuple(temp_list[i]), 
                    depth = self.depth + 1, 
                    id = self.id + [i], 
                    settings_dict = self.settings_dict,
                    img_array = self.img_array
                    ) # note convert (see above)

            # recursively define true (if iter)
            # if iter and self.depth + 1 < self.settings_dict['max_depth'] and self._calculate_std() < self.settings_dict['std_delta']: # and pixel variability...
            if iter:
                curr_node._divide(iter = iter)

            self.daughter_nodes.append(
                curr_node
                ) 

        self.is_leaf = False

    def _calculate_img_statistics(self):
        # in progress...
        # if self._slice_img() is None:
        #     return np.inf
        if self.img_array is None:
            return 0, np.inf
        else:
            sliced_img = slice_img(self.img_array, self)
            return np.mean(sliced_img), np.std(sliced_img)

    def __check_tuple(self, inds):
        """ __check_tuple
        Input tuple must be checked for correct "formatting".
        """

        assert isinstance(inds, tuple)


        for x in inds:
            assert isinstance(x, tuple)
            assert len(x) == 2
            assert isinstance(x[0], int) and isinstance(x[1], int)
            assert x[1] > x[0] >= 0

        # ...

        return True

def slice_img(img_array, node):
        assert isinstance(node, Node)
        slices = [slice(node[i][0], node[i][1]) for i in range(node.dim)]
        return img_array[*slices]

def leaf_vec_to_img(qtree, vec):
    assert len(vec) == len(qtree.leaf_list)

    ret_img = np.zeros(np.shape(qtree.img_array))

    for i in range(len(vec)):
        set_img(ret_img, qtree.leaf_list[i], vec[i])
    
    return ret_img


def set_img(img_array, node, val):
    assert isinstance(node, Node)
    slices = [slice(node[i][0], node[i][1]) for i in range(node.dim)]
    img_array[*slices] = val
    # return img_array


class QTree2():

    # def __new__ (cls, img_array, **kwargs):
    #     sh = img_array.shape
    #     inds = tuple([(0, sh_i) for sh_i in sh])

    #     assert isinstance(inds, tuple)
    #     return super(QTree, cls).__new__(cls, inds, **kwargs)

    def __init__(
            self, 
            img_array,
            min_depth = 0,
            max_depth = 6, 
            std_delta = 4
            ):
        
        # initialise inds from img_array
        sh = img_array.shape
        inds = tuple([(0, sh_i) for sh_i in sh])
            
        self.node_tree = Node(
            inds,
            depth = 0,
            img_array=img_array
        )

        self.img_array = img_array

        self.node_tree.set_settings(min_depth=min_depth, max_depth=max_depth, std_delta=std_delta)
        # print(tmp)

        # super().__init__(
        #     inds,
        #     depth = 0,
        #     max_depth=max_depth, 
        #     std_thresh=std_thresh,
        #     img_array=img_array
        #     )
        
        self.node_tree._divide(iter=True)

        self.leaf_list = self._traverse(self.node_tree)

        # make compressed img
        self.cimg_array = np.zeros(np.shape(self.img_array))
        self.leaf_val = []
        for leaf in self.leaf_list:
            val = leaf.img_avg
            set_img(self.cimg_array, leaf, val)
            self.leaf_val.append(val)

        self.cr = len(self.leaf_list)/np.size(self.img_array)

        self._get_leaf_inc()


        # self.load_array(img_array)

    # def load_array(self, img_array):
    #     # Checks...
    #     # numpy array.

    #     self.img_array = img_array
    #     self.sh = self.img_array.shape

    def _get_leaf_inc(self):
        """
            _get_leaf_inc
        Get an inclusion map from the image array structure to location in the leaf list.
        Useful for defining partitions on the image array.

        # may be redundant method; we could combine with cimg_array construction?
        """
        self.leaf_inc = np.zeros(np.shape(self.img_array), dtype=int)
        for i in range(len(self.leaf_list)):
            set_img(self.leaf_inc, self.leaf_list[i], i)
        
    
            

    def _traverse(self, node):

        if node.is_leaf:
            # print(node.id)
            return [node]
        else:
            tmp = []
            for x in node.daughter_nodes:
                tmp = tmp + self._traverse(x)
            return tmp

        


# Is this needed???
class Pixel():

    def __init__(self, coords, val):
        """ Pixel
        Stores information about a specific pixel and its relation to the entire qtree structure.

        Do we really need this? Would get quite large for ND images...
        """

        # assert isinstance(coords, tuple)

        self.coords = coords
        self.val = val

        self.node_id = ''
        self.node_id_assigned = False

        self.node_val = None # this gets averaged over pixels in the node.

    def assign_to_node(self, node_label, node_val):
        """ assign_to_node
        Assign Pixel to a node in the qtree.
        """
        self.node_id = node_label
        self.node_id_assigned = True
        self.node_val = node_val