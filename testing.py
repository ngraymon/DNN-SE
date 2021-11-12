import torch
import numpy as np

from FNN import *

model = FermiNet(5, 1, np.array([[1,1,1], [-1,1,1]]), np.array([[0,2,1], [0,0,1]]), num_determinants=2)

wavefn = model.forward()
print(wavefn)