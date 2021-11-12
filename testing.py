import numpy as np

from FNN import FermiNet

#must give electron/nuclei positions as numpy arrays
#here is example with 5 layers, 2 electrons (1up, 1down), 2 nuclei, 2 final determinants, and default layer sizing (custom_h_sizes=false)
model = FermiNet(5, 1, np.array([[1,1,1], [-1,1,1]]), np.array([[0,2,1], [0,0,1]]), num_determinants=2)

wavefn = model.forward()
print(wavefn)