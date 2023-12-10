# %%
# This is a replica of the simulation proposed in "Diffusion tensor of water in model articular cartilage" by Konstantin I. Momot, Eur Biophys J (2011) 40:81–91,
# DOI 10.1007/s00249-010-0629-4
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import multiprocessing
import copy
import Fibril
import GetIntersect
import DetailedPath

N_p = 400  # number of particles
N_t = 30000  # number of time steps
dt = 5e-9  # s
D_0 = 2.3e-3  ## mm^2/s
dr = np.sqrt(6 * D_0 * dt)  # mm (order of 10^-6)
L_0 = 1e-4  ##mm
T = []  # needs to be correlated to phi (volume fibrils/total volume) via computephi
for i in range(0, 15):
    T.append(i * 0.1)
T.append(193 / 120)
Phis = []
# eigenvalues are ordered on the value of the scal prod of eigenvector and k versor
D_1 = []  # principal eigenvalues of DT matrix
D_23 = []  # mean between secondary and tertiary eigenvalues


def ComputePhi(T, points):
    accepted = 0
    for i in range(int(-points / 2), int(points / 2)):
        for j in range(int(-points / 2), int(points / 2)):
            if Fibril.Fibril(i * 1e-7, j * 1e-7, L_0) > T:
                accepted += 1
    return accepted / ((2 * int(points / 2)) ** 2)


# updates the coordinates by uniformily sampling the polar angles (dr is fixed by D_0 and dt)


# THIS IS WELL WRITTEN, DO NOT TOUCH
def Boost(coordinates, dr):
    theta = 2 * np.pi * (np.random.uniform())
    phi = np.arccos(2 * np.random.uniform() - 1)
    coordinates[0] += dr * np.sin(phi) * np.cos(theta)
    coordinates[1] += dr * np.sin(phi) * np.sin(theta)
    coordinates[2] += dr * np.cos(phi)
    return phi, theta


# returns an addend of an element of the diffusion tensor, i and j are the components of the coordinates (0=x,1=y,2=z)


def GetDTensorElementAddend(coordinateshistory, i, j):
    return (coordinateshistory[1][i] - coordinateshistory[0][i]) * (
        coordinateshistory[1][j] - coordinateshistory[0][j]
    )


# simulates one particle


def SimulParticle(particle_index, N_t, N_p, dt, dr, L_0, T):
    np.random.seed()
    #if particle_index % 20 == 0:
    #    print(particle_index)
    coordinates = [
        L_0*np.random.uniform(0, 1),
        L_0*np.random.uniform(0, 1),
        L_0*np.random.uniform(0, 1),
    ]
    while Fibril.Fibril(coordinates[0], coordinates[1], L_0) - T > 0:
        coordinates = [
            L_0*np.random.uniform(0, 1),
            L_0*np.random.uniform(0, 1),
            L_0*np.random.uniform(0, 1),
        ]
    coordinateshistory = [copy.deepcopy(coordinates), []]
    previous_border_interaction = 0
    for i in range(0, N_t):
        if previous_border_interaction == 0 :
            oldcoord = copy.deepcopy(coordinates)
            phi, theta = Boost(coordinates, dr)
            if Fibril.Fibril(coordinates[0], coordinates[1], L_0) - T > 0 :
                coordinates = GetIntersect.GetIntersect(
                    coordinates, oldcoord, phi, theta, T, L_0
                )
                previous_border_interaction = 1
        if previous_border_interaction == 1 :
            oldcoord = copy.deepcopy(coordinates)
            phi, theta = Boost(coordinates, dr)
            condition = DetailedPath.DetailedPath(coordinates, oldcoord, T, L_0)
            if condition == True :
               coordinates = copy.deepcopy(oldcoord)
            else:
               previous_border_interaction = 0
    coordinateshistory[1] = copy.deepcopy(coordinates)
    D_xx = GetDTensorElementAddend(coordinateshistory, 0, 0) / (2 * N_t * dt * N_p)
    D_xy = GetDTensorElementAddend(coordinateshistory, 0, 1) / (2 * N_t * dt * N_p)
    D_xz = GetDTensorElementAddend(coordinateshistory, 0, 2) / (2 * N_t * dt * N_p)
    D_yy = GetDTensorElementAddend(coordinateshistory, 1, 1) / (2 * N_t * dt * N_p)
    D_yz = GetDTensorElementAddend(coordinateshistory, 1, 2) / (2 * N_t * dt * N_p)
    D_zz = GetDTensorElementAddend(coordinateshistory, 2, 2) / (2 * N_t * dt * N_p)
    # the tensor is symm.
    return [[D_xx, D_xy, D_xz], [D_xy, D_yy, D_yz], [D_xz, D_yz, D_zz]]


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
        mean_sec_eval = (sum(evalues)-principal_eval) / 2
        if np.abs(evectors[2][1]) > np.abs(evectors[2][0]):
            principal_eval = evalues[1]
            mean_sec_eval = (sum(evalues)-principal_eval) / 2
            if np.abs(evectors[2][2]) > np.abs(evectors[2][1]):
                principal_eval = evalues[2]
                mean_sec_eval = (sum(evalues)-principal_eval) / 2
        elif np.abs(evectors[2][2]) > np.abs(evectors[2][0]) :
            principal_eval = evalues[2]
            mean_sec_eval = (sum(evalues)-principal_eval) / 2
        D_1.append(principal_eval)
        D_23.append(mean_sec_eval)
        print(t)
        print(DT[0])
        print(DT[1])
        print(DT[2])

    # compute phis
    for t in T:
        Phis.append(ComputePhi(t, 1000))
    # plotting
    plt.scatter(Phis, D_1,marker='x')
    plt.scatter(Phis, D_23)
# %%
