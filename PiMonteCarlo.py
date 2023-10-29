# %%
import numpy as np
import matplotlib.pyplot as plt

accepted = 0
total = 10000000

for i in range(1, total + 1):
    x = np.random.random()
    y = np.random.random()
    dist = np.sqrt(x**2 + y**2)
    if dist <= 1:
        accepted += 1
    #       plt.xlim(0,1)
    #       plt.ylim(0,1)
    #       plt.scatter(x,y,color="green")
    #   else:
    #      plt.xlim(0,1)
    #      plt.ylim(0,1)
    #      plt.scatter(x,y,color="red")
    pi = 4 * accepted / i
    err = np.pi - pi
    print("N = %d; pi = %5.10f; err = %5.2e" % (i, pi, err))
# %%
