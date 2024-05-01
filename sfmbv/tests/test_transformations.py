
import unittest
from sfmbv import methods
import numpy as np

def _gen(n, k = 2):
    # Generate a random multi-dimensional array.
    return np.random.rand(*[k for _ in range(n)])

class TestTransformations(unittest.TestCase):

    def test_transformations(self, N = 4):
        """
            We map multi-dimensional image arrays to nodes of an ImageGraph and back to 
            image space to ensure that the transformations are invertible.
        """
        
        # NOTE: Code does not handle the trivial n=1 image case.
        for n in range(1,N+1):
            img_array = _gen(n)

            G = methods.ImageGraph(img_array)
            node_vec = methods.img_array_to_node_vec(G, img_array)

            for i in range(len(node_vec)):
                # Check that node_vec is ordered as expected.
                assert node_vec[i] == img_array[*G.v_coords[G.list_nodes[i]]]
            
            img_reconstructed_array = methods.node_vec_to_img_array(G, node_vec)

            # Check that through the whole transformation process we can reconstruct the original array.
            assert (img_array == img_reconstructed_array).all() 

        def test_transformations_with_mask(self, N = 3):
            """
                ... there seems to be some issues with masking and reconstruction. Note sure what's the issue yet. :-(
            """
            0

if __name__ == '__main__':
    unittest.main()