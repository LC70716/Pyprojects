#%%
import numpy as np
import matplotlib.pyplot as plt
#note that additional optimization may be performed by sampling directly the cosine and sine of random variables when calculating the "boost" function
#note that particles DO NOT interact w/ each other 

N_p = 100 #number of particles
N_t = 100 #number of time steps
dt = 5e-9 ##s
D_0 = 2.3e-3 ## mm^2/s
dr = np.sqrt(6*D_0*dt) #mm
L_0 = 1e-4 ##mm
T=0.75 #nedds to be correlated to phi (volume fibrils/total volume) via computephi
coordinates = [0,0,0] #coordinates of the simulated particle
coordinateshistory = [[0,0,0],[0,0,0]] #only stores initial and finale position,useful to get the tensor components
DT = [[0,0,0]
      [0,0,0]
      [0,0,0]] 

#eigenvalues are ordered on the value of the scal prod of eigenvector and k versor
D_1 = [] #principal eigenvalues matrix
D_2 = [] #secondary eigenvalues
D_3 = [] #tertiary eigenvalues

#this is very expensive, but is only a test to visualize the fibrils
#for a simulation, this would need to be calculated about N_p*N_t, going by the paper's numbers
# this would be computed (about 1e5) * (about 1e4) = 1e9 times, which on my machine takes about 1 m

def Fibril(x,y,L_0) :
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

#cannot properly verify numerically,only visually on the plot,seems to work

def ComputePhi(T,points) :
    accepted = 0
    for i in range(0,points) :
      for j in range(0,points):
        if fibril(i*1e-7,j*1e-7,L_0) > T :
            accepted += 1
    return accepted/(points**2) 

#updates the coordinates by uniformily sampling the polar angles (dr is fixed by D_0 and dt)

def Boost(coordinates,dr) :
    phi = np.random.uniform(0,np.pi) 
    theta = np.random.uniform(0,2*np.pi)
    coordinates[0] += dr*np.sin(phi)*np.cos(theta)
    coordinates[1] += dr*np.sin(phi)*np.sin(theta)
    coordinates[2] += dr*np.cos(phi)

#returns an addend of an element of the diffusion tensor, i and j are the components of the coordinates (0=x,1=y,2=z)

def GetDTensorElementAddend(coordinateshistory,i,j,N_t,N_p,dt) :
    return (coordinateshistory[1][i]-coordinateshistory[0][i])*(coordinateshistory[1][j]-coordinateshistory[0][j])/(2*N_t*dt*N_p)
     