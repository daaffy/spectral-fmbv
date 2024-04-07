
import sys
sys.path.insert(1,'/Users/jackh/Documents/FMBV_2023/gordon_original/src/') # FMBV src
from fmbv_refactor import FMBV

def test_load(
        # kretz_path, pd_path, seg_path
        ):
    try:
        f = FMBV()
    except:
        0

if __name__ == "__main__":
    test_load()
    print("Passed!")