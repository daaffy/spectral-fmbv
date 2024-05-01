import unittest
from sfmbv import methods
import numpy as np

def _gen(n, k = 2, MASK_THRESHOLD=.8):
    # Generate a random multi-dimensional array and corresponding mask.
    return np.random.rand(*[k for _ in range(n)]), np.random.rand(*[k for _ in range(n)]) < MASK_THRESHOLD

class TestImageArray(unittest.TestCase):

    def test_masking(self, LOOP_NUM=10, N=3):
        """
            Ensure that masking in image space produces the correct "masked graph."
        """
        # Loop dimensions.
        for n in range(1,N+1):

            # Loop for randomness.
            for _ in range(LOOP_NUM):
                img_array, mask_array = _gen(n)

                # masked_values = img_array[mask_array]

                G = methods.ImageGraph(img_array, mask_array=mask_array) # No zero_node here (-1).

                # Check through list of nodes and ensure that they all belong to the mask.
                for ni in G.list_nodes:
                    assert not ni == -1 and mask_array[*G.v_coords[ni,:]]

                node_vec = methods.img_array_to_node_vec(G, mask_array)
                assert (node_vec == 1.).all() # Check that mask array is mapped to the nodes.

    def test_coarse(self):
        # construct a coarsening class on an image
        # build coarse graph
        # test transformations
        # ...
        # test with masking
        0

if __name__ == '__main__':
    unittest.main()