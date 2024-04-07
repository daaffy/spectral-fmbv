import sys
sys.path.insert(1,sys.path[0]+'/../src')

import methods
import numpy as np

G = methods.ImageGraphTest()
F = methods.ImageGraph(np.zeros((2,2)))

print(G.edges)



