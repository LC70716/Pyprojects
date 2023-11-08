# %%
# This is a replica of the simulation proposed in "Diffusion tensor of water in model articular cartilage" by Konstantin I. Momot, Eur Biophys J (2011) 40:81–91,
# DOI 10.1007/s00249-010-0629-4
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# note that additional optimization may be performed by sampling directly the cosine and sine of random variables when calculating the "boost" function
# note that particles DO NOT interact w/ each other

N_p = 10000  # number of particles
N_t = 30000  # number of time steps
dt = 5e-9  ##s
D_0 = 2.3e-3  ## mm^2/s
dr = np.sqrt(6 * D_0 * dt)
# mm (order of 10^-6)
# about two orders of magnitude lower than the radius of the fibrils : the case in which a particle would pass inside a fibril but end outside
# are therefore not treated since this effect is assumed to be negligible
L_0 = 1e-4  ##mm
T = []
Phis = []
for i in range(0, 17):
    T.append(i * 0.1)
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
    print("done")
    return accepted / ((2 * int(points / 2)) ** 2)


# updates the coordinates by uniformily sampling the polar angles (dr is fixed by D_0 and dt)


def Boost(coordinates, dr):
    phi = np.random.uniform(0, np.pi)
    theta = np.random.uniform(0, 2 * np.pi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    coordinates[0] += dr * sin_phi * np.sqrt(1 - sin_theta**2)
    coordinates[1] += dr * sin_phi * sin_theta
    coordinates[2] += dr * np.sqrt(1 - sin_phi**2)


# returns a vector to be subtracted from the coordinates when a particle would fall into a fibril (ie Fibril(x,y,L_0) - T >= 0),dist is the distance
# between where the particle would end up and the boundary : this is equal to Firbril(x,y,L_0) - T


def GetBorderPos(oldcoord, insidecoord, dist):
    theta = np.arctan(
        (insidecoord[1] - oldcoord[1]) / (insidecoord[0] - insidecoord[1])
    )  # angle between particle trajectory and x-axis
    return [
        oldcoord[0] - dist * np.cos(theta),
        oldcoord[1] - dist * np.sin(theta),
        oldcoord[2],
    ]


# returns an addend of an element of the diffusion tensor, i and j are the components of the coordinates (0=x,1=y,2=z)


def GetDTensorElementAddend(coordinateshistory, i, j, N_t, N_p, dt):
    return (
        (coordinateshistory[1][i] - coordinateshistory[0][i])
        * (coordinateshistory[1][j] - coordinateshistory[0][j])
        / (2 * N_t * dt * N_p)
    )


def MainLoop(N_t, N_p, dt, dr, L_0, T):
    coordinates = [
        np.random.uniform(-1e-3, 1e-3),
        np.random.uniform(-1e-3, 1e-3),
        np.random.uniform(-1e-3, 1e-3),
    ]
    while Fibril(coordinates[0], coordinates[1], L_0) - T > 0:
        coordinates = [
            np.random.uniform(-1e-3, 1e-3),
            np.random.uniform(-1e-3, 1e-3),
            np.random.uniform(-1e-3, 1e-3),
        ]
    coordinateshistory = [[0, 0, 0], [0, 0, 0]]
    coordinateshistory[0] = [coordinates[0], coordinates[1], coordinates[2]]
    for i in range(0, N_t):
        oldcoord = coordinates
        Boost(coordinates, dr)
        bordercrossing = Fibril(coordinates[0], coordinates[1], L_0)
        if bordercrossing - T > 0:
            coordinates = GetBorderPos(oldcoord, coordinates, bordercrossing - T)
    coordinateshistory[1] = [coordinates[0], coordinates[1], coordinates[2]]
    D_xx = GetDTensorElementAddend(coordinateshistory, 0, 0, N_t, N_p, dt)
    D_xy = GetDTensorElementAddend(coordinateshistory, 0, 1, N_t, N_p, dt)
    D_xz = GetDTensorElementAddend(coordinateshistory, 0, 2, N_t, N_p, dt)
    D_yy = GetDTensorElementAddend(coordinateshistory, 1, 1, N_t, N_p, dt)
    D_yz = GetDTensorElementAddend(coordinateshistory, 1, 2, N_t, N_p, dt)
    D_zz = GetDTensorElementAddend(coordinateshistory, 2, 2, N_t, N_p, dt)
    # the tensor is symm.
    return [[D_xx, D_xy, D_xz], [D_xy, D_yy, D_yz], [D_xz, D_yz, D_zz]]


# simulation start
for t in T:
    princ_evals = []
    sec_vals = []
    DT = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(0, N_p):
        result = MainLoop(N_t, N_p, dt, dr, L_0, t)
        DT = [
            [DT[i][j] + result[i][j] for j in range(len(DT[0]))] for i in range(len(DT))
        ]
    w, v = np.linalg.eig(DT)
    print("e-values:", w)
    print("e-vectors:", v)
    principal_eval = w[0]
    mean_sec_eval = (w[1] + w[2]) / 2
    if np.abs(v[1][2]) > np.abs(v[0][2]):
        principal_eval = w[1]
        mean_sec_eval = (w[0] + w[2]) / 2
        if np.abs(v[2][2]) > np.abs(v[1][2]):
            principal_eval = w[2]
            mean_sec_eval = (w[0] + w[1]) / 2
    else:
        if np.abs(v[2][2]) > np.abs(v[0][2]):
            principal_eval = w[2]
            mean_sec_eval = (w[0] + w[1]) / 2
    D_1.append(principal_eval)
    D_23.append(mean_sec_eval)
    print(t)

# compute phis
for t in T:
    Phis.append(ComputePhi(t, 1000))

# plotting
plt.scatter(Phis, D_1)
# %%
