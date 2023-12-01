# %%
# This is a replica of the simulation proposed in "Diffusion tensor of water in model articular cartilage" by Konstantin I. Momot, Eur Biophys J (2011) 40:81–91,
# DOI 10.1007/s00249-010-0629-4
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import multiprocessing
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# note that additional optimization may be performed by sampling directly the cosine and sine of random variables when calculating the "boost" function
# note that particles DO NOT interact w/ each other

N_p = 400  # number of particles
N_t = 30000  # number of time steps
dt = 5e-9  ##s
D_0 = 2.3e-3  ## mm^2/s
dr = np.sqrt(6 * D_0 * dt)
# mm (order of 10^-6)
# about two orders of magnitude lower than the radius of the fibrils : the case in which a particle would pass inside a fibril but end outside
# are therefore not treated since this effect is assumed to be negligible
L_0 = 1e-4  ##mm
T = []
T.append(0)
Phis = []
#for i in range(0, 17):
  #  T.append(i * 0.1)
#T.append(193 / 120)
# needs to be correlated to phi (volume fibrils/total volume) via computephi


# eigenvalues are ordered on the value of the scal prod of eigenvector and k versor
D_1 = []  # principal eigenvalues of DT matrix
D_23 = []  # mean between secondary and tertiary eigenvalues

# this is expensivish
# for a simulation, this would need to be calculated about N_p*N_t, going by the paper's numbers
# this would be computed (about 1e5) * (about 1e4) = 1e9 times, which on my machine takes about 1 m


def Fibril(x, y, L_0):
    argx = 2 * np.pi * x / L_0
    argy = 2 * np.pi * y / L_0
    term1 = np.cos(argx)
    term2 = (81 / 320) * (term1**2)
    term3 = (1 / 15) * (term1**3)
    term4 = (3 / 320) * (term1**4)
    term5 = np.cos(argy)
    term6 = (81 / 320) * (term5**2)
    term7 = (1 / 15) * (term5**3)
    term8 = (3 / 320) * (term5**4)
    return term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8


# cannot properly verify numerically,only visually on the plot,seems to work


def ComputePhi(T, points):
    accepted = 0
    for i in range(int(-points / 2), int(points / 2)):
        for j in range(int(-points / 2), int(points / 2)):
            if Fibril(i * 1e-7, j * 1e-7, L_0) > T:
                accepted += 1
    return accepted / ((2 * int(points / 2)) ** 2)


# updates the coordinates by uniformily sampling the polar angles (dr is fixed by D_0 and dt)

#THIS IS WELL WRITTEN, DO NOT TOUCH
def Boost(coordinates, dr):
    theta = 2*np.pi*(np.random.uniform())
    phi = np.arccos(2*np.random.uniform() - 1)
    coordinates[0] += dr * np.sin(phi) * np.cos(theta)
    coordinates[1] += dr * np.sin(phi) * np.sin(theta)
    coordinates[2] += dr * np.cos(phi)
    return phi, theta


# returns a vector to be subtracted from the coordinates when a particle would fall into a fibril (ie Fibril(x,y,L_0) - T >= 0),dist is the distance
# between where the particle would end up and the boundary : this is equal to Firbril(x,y,L_0) - T
# NEED TO FIND A WAY TO CALCULATE INTERCEPTION POINT WITH FIBRIL

def GetBorderPos(insidecoord, dist, phi, theta):
        insidecoord[0] -= dist * np.sin(theta) * np.cos(phi)
        insidecoord[1] -= dist * np.sin(theta) * np.sin(phi)
        insidecoord[2] -= dist * np.cos(theta)

# returns an addend of an element of the diffusion tensor, i and j are the components of the coordinates (0=x,1=y,2=z)

def GetDTensorElementAddend(coordinateshistory, i, j):
    return (coordinateshistory[1][i] - coordinateshistory[0][i]) * (coordinateshistory[1][j] - coordinateshistory[0][j])

# simulates one particle

def SimulParticle(particle_index, N_t, N_p, dt, dr, L_0, T):
    np.random.seed()
    coordinates = [
        np.random.uniform(0, L_0),
        np.random.uniform(0, L_0),
        np.random.uniform(0, L_0),
    ]
    #while Fibril(coordinates[0], coordinates[1], L_0) - T > 0:
    #    coordinates = [
    #       np.random.uniform(-1, 1),
    #        np.random.uniform(-1, 1),
    #        np.random.uniform(-1, 1),
    #    ]
    coordinateshistory = [[coordinates[0], coordinates[1], coordinates[2]], []]
    for i in range(0, N_t):
        theta, phi = Boost(coordinates, dr)
        #bordercrossing = Fibril(coordinates[0], coordinates[1], L_0)
        #while bordercrossing - T > 0 :  # MUST PROPERLY EVALUATE DIST
        #    dist =  dr/10  # work with this for the moment
        #    GetBorderPos(coordinates, dist, theta, phi)
        #    bordercrossing = Fibril(coordinates[0], coordinates[1], L_0)
    #coordinates is now updated
    coordinateshistory[1] = [coordinates[0], coordinates[1], coordinates[2]]
    D_xx = GetDTensorElementAddend(coordinateshistory, 0, 0)/ (2 * N_t * dt * N_p)
    D_xy = GetDTensorElementAddend(coordinateshistory, 0, 1)/ (2 * N_t * dt * N_p)
    D_xz = GetDTensorElementAddend(coordinateshistory, 0, 2)/ (2 * N_t * dt * N_p)
    D_yy = GetDTensorElementAddend(coordinateshistory, 1, 1)/ (2 * N_t * dt * N_p)
    D_yz = GetDTensorElementAddend(coordinateshistory, 1, 2)/ (2 * N_t * dt * N_p)
    D_zz = GetDTensorElementAddend(coordinateshistory, 2, 2)/ (2 * N_t * dt * N_p)
    # the tensor is symm.
    return [[D_xx, D_xy, D_xz], 
            [D_xy, D_yy, D_yz], 
            [D_xz, D_yz, D_zz]]


def MainLoop(N_t, N_p, dt, dr, L_0, T):
    pool = multiprocessing.Pool()  # Create a pool of worker processes
    results = []
    # Mapping the SimulParticle function to all particle indices in parallel
    for i in range(N_p):
        result = pool.apply_async(SimulParticle, args=(i, N_t, N_p, dt, dr, L_0, T))
        results.append(result)
    # Getting the results for all particles
    particle_results = [result.get() for result in results]
    # Closing the pool and waiting for all processes to finish
    pool.close()
    pool.join()
    return particle_results


# simulation start
if __name__ == "__main__":
    for t in T:
        DT = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        sim_results = MainLoop(N_t, N_p, dt, dr, L_0, t)
        for result in sim_results:
            DT = [
                [DT[i][j] + result[i][j] for j in range(len(DT[0]))]
                for i in range(len(DT))
            ]
        print(DT)
        evalues, evectors = np.linalg.eig(DT)
        print("e-values:", evalues)
        print("e-vectors:", evectors)
        principal_eval = evalues[0]
        mean_sec_eval = (evalues[1] + evalues[2]) / 2
        if np.abs(evectors[2][1]) > np.abs(evectors[2][0]):
            principal_eval = evalues[1]
            mean_sec_eval = (evalues[0] + evalues[2]) / 2
            if np.abs(evectors[2][2]) > np.abs(evectors[2][1]):
                principal_eval = evalues[2]
                mean_sec_eval = (evalues[0] + evalues[1]) / 2
        else:
            if np.abs(evectors[2][2]) > np.abs(evectors[2][0]):
                principal_eval = evalues[2]
                mean_sec_eval = (evalues[0] + evalues[1]) / 2
        D_1.append(principal_eval)
        D_23.append(mean_sec_eval)
        print(t)

    # compute phis
    #for t in T:
    #   Phis.append(ComputePhi(t, 1000))
# plotting
#plt.scatter(Phis, D_1)
#plt.scatter(Phis, D_23)
# %%