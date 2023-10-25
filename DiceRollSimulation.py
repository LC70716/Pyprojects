# %%
import numpy as np
import matplotlib.pyplot as plt

N=2000
rolls = [0,0,0,0,0,0] #or [0]*6
x = [i for i in range(1,7)]
for i in range(0,N) : 
  r = np.random.randint(1,7)
  rolls[r-1] += 1
  plt.ylim(0,N*2/3)
  plt.bar(x,rolls)
  plt.hlines(N/6,0.5,6.5,color = "red")


# %%
