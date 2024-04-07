import sys
sys.path.insert(1,'/Users/jackh/Documents/FMBV_2023/gordon_original/src/') # FMBV src

import numpy as np
from StructuralFMBV import StructuralFMBV
import unittest

pd_path = "imgs/test_pd.nii.gz"
seg_path = "imgs/test_seg.nii.gz"
kretz_path = "imgs/test_kretz.vol"
fake_path = "$$$"

class TestLoad(unittest.TestCase):
    def test_load(self):
        sf = StructuralFMBV()
        sf.load_pd(pd_path)
        sf.load_seg(seg_path)
        sf.load_kretz(kretz_path)

class TestLaplacian(unittest.TestCase):
    0

if __name__ == "__main__":
    unittest.main()