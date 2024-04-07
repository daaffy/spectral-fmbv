
class Points:

    def __init__(self, img_array):
        # not sure about this yet...
        0

class Point:

    def __init__(self, coords, val):
        '''
            Point
        Point data-structure to store pixel indexes, pixel values, and resulting node membership for inclusion map.

        '''
        self.coords = coords
        self.val = val

        self.node_id = ''
        self.node_id_assigned = False

        self.node_val = None # this gets averaged over pixels in the node.

    def assign_node(self, node_label, node_val):
        self.node_id = node_label
        self.node_id_assigned = True
        self.node_val = node_val
