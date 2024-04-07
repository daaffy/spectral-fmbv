
import sys
sys.path.insert(1,'/Users/jackh/Documents/FMBV_2023/gordon_original/src/') # FMBV src

import numpy as np
from fmbv_refactor import FMBV
import unittest

pd_path = "imgs/test_pd.nii.gz"
seg_path = "imgs/test_seg.nii.gz"
kretz_path = "imgs/test_kretz.vol"
fake_path = "$$$"

MM = False

def load_default(**kwargs):
    f = FMBV(**kwargs)
    f.load_pd(pd_path)
    f.load_seg(seg_path)
    f.load_kretz(kretz_path)

    return f

# @unittest.SkipTest
class TestLoad(unittest.TestCase):
    def test_initialise(self):
        f = FMBV()

    def test_load(self):
        f = FMBV()

        f.load_pd(pd_path)
        assert f.pd_supplied == True
        assert f.seg_supplied == False
        assert f.kretz_supplied == False

        with self.assertRaises(Exception):
            f._check_pd_seg_sizes()

        f.load_seg(seg_path)
        assert f.seg_supplied == True

        f._check_pd_seg_sizes()

        with self.assertRaises(Exception):
            f.get_distance_map(fake_path)

        f.load_kretz(kretz_path)
        assert f.kretz_supplied == True

        # Load an non-existing file
        g = FMBV()

        with self.assertRaises(Exception):
            g.load_pd(fake_path)

    def test_misc(self):
        # Test some pre-processing operations
        f = FMBV()

        with self.assertRaises(Exception):
            f._set_default_segmentation()

        with self.assertRaises(Exception):
            f._clean_segmentation()

        f.load_pd(pd_path)
        f._set_default_segmentation()

        f.load_seg(seg_path)
        f._clean_segmentation()

# @unittest.SkipTest
class TestGlobal(unittest.TestCase):
    def test_global(self, def_mode = 0):

        with self.assertRaises(Exception):
            f = load_default(mode = -1, verbose = False)
            f.global_method()

        f = load_default(mode = def_mode, verbose = False)
        assert f.global_figdata_std_1 == None
        f.global_method()
        assert f.global_figdata_std_1["mode"] == def_mode
        assert f.global_figdata_std_1["complete"] == True
        assert f.global_figdata_std_2["complete"] == True

        # f._std_method_0(100*np.ones(100).flatten().astype(int))

# @unittest.SkipTest     
class TestDepthCorrection(unittest.TestCase):
    def test_depth_correction(self):
        temp_path = './src/test/temp.nii.gz'

        f = load_default(mode = 0, verbose = True, mm=MM)
        f.depth_corrected_method()

        # Export and load
        if not MM:
            f.export_standardised(temp_path)

        g = load_default(mode = 0, verbose = True)
        g.load_pd(temp_path)
        g.global_method(skip_standardisation=True)

class TestStandardMethod1(unittest.TestCase):
    def test_std_method_1(self):
        f = load_default(verbose = True, mode = 1)

        pd_flat = f.pd_array.flatten()
        f._std_method_1(pd_flat)

        a, b, c, d = f.std_method(pd_flat)
        print(b)
        print(c)

if __name__ == "__main__":
    unittest.main()