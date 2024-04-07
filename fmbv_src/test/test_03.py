
import sys
sys.path.insert(1,'/Users/jackh/Documents/FMBV_2023/gordon_original/src/') # FMBV src
sys.path.insert(1,'/Users/jackh/Documents/FMBV_2023/gordon_original/') # FMBV src


import numpy as np
from fmbv_refactor import FMBV
import load
import unittest

pd_path = "imgs/test_pd.nii.gz"
seg_path = "imgs/test_seg.nii.gz"
kretz_path = "imgs/test_kretz.vol"
fake_path = "$$$"

class TestExamples(unittest.TestCase):
    def test_example_01(self):
        '''
            test_example_01
        Test of load front-end.
        '''

        # ; no kretz, no segmentation
        f = FMBV()
        f.load_pd(pd_path)
        f.load_seg('')
        # f._set_default_segmentation(mode="blank")
        f.global_method()
        f.global_fmbv_value_2

        load.run_paths(pd_path=pd_path, mode=1) # equivalent front-end

        # ; no kretz
        f = FMBV()
        f.load_pd(pd_path)
        f.load_seg(seg_path)
        f.global_method()
        f.global_fmbv_value_2

        load.run_paths(pd_path=pd_path, seg_path=seg_path, mode=1) # equivalent front-end

        load.run_paths(pd_path=pd_path, seg_path=seg_path, kretz_path=kretz_path, mode=1) # equivalent front-end


if __name__ == "__main__":
    unittest.main()