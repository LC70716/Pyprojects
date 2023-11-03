#%%
import numpy as np
import matplotlib.pyplot as plt

N_p = 100 #number of particles
N_t = 100 #number of time steps
dt = 5e-9 ##s
D_0 = 2.3e-3 ## mm^2/s
dr = np.sqrt(6*D_0*dt) #mm
L_0 = 1e-4 ##mm
T=1

def fibril(x,y,L_0) :
    argx = 2*np.pi*x/L_0
    argy = 2*np.pi*y/L_0
    term1 = np.cos(argx)
    term2 = (81/320)*((np.cos(argx))**2)
    term3 = (1/15)*((np.cos(argx))**3)
    term4 = (3/320)*((np.cos(argx))**4)
    term5 = np.cos(argy)
    term6 = (81/320)*((np.cos(argy))**2)
    term7 = (1/15)*((np.cos(argy))**3)
    term8 = (3/320)*((np.cos(argy))**4)
    return term1-term2+term3-term4+term5-term6+term7-term8

#this is very expensive, but is only a test to visualize the fibrils
#for a simulation, this would need to be caluclated about N_p*N_t, going by the paper's numbers
# this would be computed (about 1e5) * (about 1e4) = 1e9 times, which on my machine takes about 1 m
for i in range(0,10000) :
    for j in range(0,1000):
        acceptance = fibril(i*1e-6,j*1e-6,L_0)
print("end")
# %%
