import DiffTensorSquareCellsMC
import numpy as np

dt = 5e-9  # s
D_0 = 2.3e-3  ## mm^2/s
dr = np.sqrt(6 * D_0 * dt)
L_0 = 1e-4  ##mm
T = []  # needs to be correlated to phi (volume fibrils/total volume) via computephi
for i in range(0, 17):
    T.append(i * 0.1)
T.append(193 / 120)
if __name__ == "__main__":
    DiffTensorSquareCellsMC.MainLoop(30000, 4000, dt, dr, L_0, T)
