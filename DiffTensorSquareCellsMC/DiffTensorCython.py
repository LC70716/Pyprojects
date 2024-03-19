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

N_p = 100  # number of particles
N_t = 100  # number of time steps, default to 30000
dt_sim = 5e-9  # s
D_0 = 2.3e-3  ## mm^2/s
D_1 = 2.0e-3
L_0 = 1e-4  ##mm
L_agg = 2.0e-3
T = [
    0,
    0.2,
    0.5,
    0.9,
    1.3,
]  # needs to be correlated to phi (volume fibrils/total volume) via computephi
Phis = []
# eigenvalues are ordered on the value of the scal prod of eigenvector and k versor
D_prin = []  # principal eigenvalues of DT matrix
D_sec = []  # mean between secondary and tertiary eigenvalues


def Getdr(D, dt):
    return np.sqrt(6 * D * dt)

def computePG_T(T,L_agg,L_0): #computes the value to add to T in order to have a PG ring around the fibrils
    term1 = (16/21)*(np.pi^2)*L_0
    term2 = -1*L_0*T
    term3 = (120/193)*(L_agg^2)/L_0
    term4 = (240/193)*L_agg*np.sqrt((16/21)*(np.pi^2)*(193/120 - T))
    term5 = -1*(16/21)*np.pi^2
    term6 = T
    return term1 + term2 + term3 + term4 + term5 + term6

def ComputePhi(T, points):
    accepted = 0
    for i in range(int(-points / 2), int(points / 2)):
        for j in range(int(-points / 2), int(points / 2)):
            if Fibril.Fibril(i * 1e-7, j * 1e-7, L_0) > T:
                accepted += 1
    return accepted / ((2 * int(points / 2)) ** 2)


# THIS IS WELL WRITTEN, DO NOT TOUCH
def Boost(coordinates, D, dt):
    theta = 2 * np.pi * (np.random.uniform())
    phi = np.arccos(2 * np.random.uniform() - 1)
    coordinates[0] += Getdr(D, dt) * np.sin(phi) * np.cos(theta)
    coordinates[1] += Getdr(D, dt) * np.sin(phi) * np.sin(theta)
    coordinates[2] += Getdr(D, dt) * np.cos(phi)
    return phi, theta


# returns an addend of an element of the diffusion tensor, i and j are the components of the coordinates (0=x,1=y,2=z)


def GetDTensorElementAddend(coordinateshistory, i, j):
    return (coordinateshistory[1][i] - coordinateshistory[0][i]) * (
        coordinateshistory[1][j] - coordinateshistory[0][j]
    )


# simulates one particle


def SimulParticle(particle_index, N_t, N_p, dt, L_0, T, PG_T):
    np.random.seed()
    counter = 0
    if particle_index % 5 == 0:
        print(particle_index)
    coordinates = [
        L_0 * np.random.uniform(0, 1),
        L_0 * np.random.uniform(0, 1),
        L_0 * np.random.uniform(0, 1),
    ]
    fibril = Fibril.Fibril(coordinates[0], coordinates[1], L_0)
    while fibril >= T:
        coordinates = [
            L_0 * np.random.uniform(0, 1),
            L_0 * np.random.uniform(0, 1),
            L_0 * np.random.uniform(0, 1),
        ]
        fibril = Fibril.Fibril(coordinates[0], coordinates[1], L_0)
    coordinateshistory = [copy.deepcopy(coordinates), []]
    previous_border_interaction = 0
    inside_PG = 0
    if fibril >= (T - PG_T):
        inside_PG = 1
    for i in range(0, N_t):
        if previous_border_interaction == 0 and inside_PG == 0:
            oldcoord = copy.deepcopy(coordinates)
            phi, theta = Boost(coordinates, D_0, dt)
            fibril = Fibril.Fibril(coordinates[0], coordinates[1], L_0)
            if fibril >= T:
                inter_coord_and_tsteps = GetIntersect.GetIntersect(
                    coordinates, oldcoord, phi, theta, T, L_0
                )
                coordinates[0] = copy.copy(inter_coord_and_tsteps[0])
                coordinates[1] = copy.copy(inter_coord_and_tsteps[1])
                coordinates[2] = copy.copy(inter_coord_and_tsteps[2])
                previous_border_interaction = 1
                inside_PG = 1
            elif fibril >= (T - PG_T):  #inside the PG ring
                inter_coord_and_tsteps = GetIntersect.GetIntersect(
                    coordinates, oldcoord, phi, theta, T - PG_T, L_0
                )
                coordinates[0] = inter_coord_and_tsteps[0] + Getdr(
                    D_1, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.sin(phi) * np.cos(theta)
                coordinates[1] = inter_coord_and_tsteps[1] + Getdr(
                    D_1, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.sin(phi) * np.sin(theta)
                coordinates[2] = inter_coord_and_tsteps[2] + Getdr(
                    D_1, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.cos(phi)
                inside_PG = 1
        elif previous_border_interaction == 0 and inside_PG == 1:
            oldcoord = copy.deepcopy(coordinates)
            phi, theta = Boost(coordinates, D_1, dt)
            fibril = Fibril.Fibril(coordinates[0], coordinates[1], L_0)
            if fibril >= T:
                inter_coord_and_tsteps = GetIntersect.GetIntersect(
                    coordinates, oldcoord, phi, theta, T, L_0
                )
                coordinates[0] = copy.copy(inter_coord_and_tsteps[0])
                coordinates[1] = copy.copy(inter_coord_and_tsteps[1])
                coordinates[2] = copy.copy(inter_coord_and_tsteps[2])
                previous_border_interaction = 1
                inside_PG = 1
            elif fibril <= (T - PG_T):  # if ends outside the PG ring
                inter_coord_and_tsteps = GetIntersect.GetIntersect(
                        coordinates, oldcoord, phi, theta, T - PG_T, L_0
                )
                coordinates[0] = inter_coord_and_tsteps[0] + Getdr(
                    D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.sin(phi) * np.cos(theta)
                coordinates[1] = inter_coord_and_tsteps[1] + Getdr(
                    D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.sin(phi) * np.sin(theta)
                coordinates[2] = inter_coord_and_tsteps[2] + Getdr(
                    D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                ) * np.cos(phi)
                inside_PG = 0
        elif previous_border_interaction == 1:
            oldcoord = copy.deepcopy(coordinates)
            phi, theta = Boost(coordinates, D_1, dt)
            border_intersection = DetailedPath.DetailedPath(
                coordinates, oldcoord, T, L_0
            )
            if border_intersection == True:
                coordinates = copy.deepcopy(oldcoord)
            else:
                fibril = Fibril.Fibril(coordinates[0], coordinates[1], L_0)
                if fibril <= (T - PG_T):  # if ends outside the PG ring
                    inter_coord_and_tsteps = GetIntersect.GetIntersect(
                        coordinates, oldcoord, phi, theta, T - PG_T, L_0
                    )
                    coordinates[0] = copy.deepcopy(inter_coord_and_tsteps[0]) + Getdr(
                        D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                    ) * np.sin(phi) * np.cos(theta)
                    coordinates[1] = copy.deepcopy(inter_coord_and_tsteps[1]) + Getdr(
                        D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                    ) * np.sin(phi) * np.sin(theta)
                    coordinates[2] = copy.deepcopy(inter_coord_and_tsteps[2]) + Getdr(
                        D_0, dt * (50 - inter_coord_and_tsteps[3]) / 50
                    ) * np.cos(phi)
                    inside_PG = 0
                    previous_border_interaction = 0
                else: # if ends inside the PG ring
                    previous_border_interaction = 0
                    inside_PG = 1
    coordinateshistory[1] = copy.deepcopy(coordinates)
    D_xx = GetDTensorElementAddend(coordinateshistory, 0, 0) / (2 * N_t * dt_sim * N_p)
    D_xy = GetDTensorElementAddend(coordinateshistory, 0, 1) / (2 * N_t * dt_sim * N_p)
    D_xz = GetDTensorElementAddend(coordinateshistory, 0, 2) / (2 * N_t * dt_sim * N_p)
    D_yy = GetDTensorElementAddend(coordinateshistory, 1, 1) / (2 * N_t * dt_sim * N_p)
    D_yz = GetDTensorElementAddend(coordinateshistory, 1, 2) / (2 * N_t * dt_sim * N_p)
    D_zz = GetDTensorElementAddend(coordinateshistory, 2, 2) / (2 * N_t * dt_sim * N_p)
    # the tensor is symm.
    return [[D_xx, D_xy, D_xz], [D_xy, D_yy, D_yz], [D_xz, D_yz, D_zz]]


def toTXT(D_prin, D_sec, Phis, filename):
    with open(filename, "w") as f:
        f.write("D_1: {}\n".format(D_prin))
        f.write("D_23: {}\n".format(D_sec))
        f.write("Phi: {}\n".format(Phis))


def MainLoop(N_t, N_p, dt, L_0, T):
    pool = multiprocessing.Pool()  # Create a pool of worker processes
    results = []
    # Mapping the SimulParticle function to all particle indices in parallel
    for i in range(N_p):
        result = pool.apply_async(SimulParticle, args=(i, N_t, N_p, dt, L_0, T))
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
        PG_T = computePG_T(t,L_agg,L_0)
        DT = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        sim_results = MainLoop(N_t, N_p, dt_sim, L_0, t, PG_T)
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
        mean_sec_eval = (sum(evalues) - principal_eval) / 2
        if np.abs(evectors[2][1]) > np.abs(evectors[2][0]):
            principal_eval = evalues[1]
            mean_sec_eval = (sum(evalues) - principal_eval) / 2
            if np.abs(evectors[2][2]) > np.abs(evectors[2][1]):
                principal_eval = evalues[2]
                mean_sec_eval = (sum(evalues) - principal_eval) / 2
        elif np.abs(evectors[2][2]) > np.abs(evectors[2][0]):
            principal_eval = evalues[2]
            mean_sec_eval = (sum(evalues) - principal_eval) / 2
        D_prin.append(principal_eval)
        D_sec.append(mean_sec_eval)
        print(t)
        print(principal_eval)
        print(mean_sec_eval)

    # compute phis
    for t in T:
        Phis.append(ComputePhi(t, 1000))
    toTXT(D_prin,D_sec,Phis,'test.txt')
# %%
