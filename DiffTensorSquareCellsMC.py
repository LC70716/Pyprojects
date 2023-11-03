# %%
# This is a replica of the simulation proposed in "Diffusion tensor of water in model articular cartilage" by Konstantin I. Momot, Eur Biophys J (2011) 40:81–91,
# DOI 10.1007/s00249-010-0629-4
import numpy as np
import matplotlib.pyplot as plt

# note that additional optimization may be performed by sampling directly the cosine and sine of random variables when calculating the "boost" function
# note that particles DO NOT interact w/ each other

N_p = 100  # number of particles
N_t = 100  # number of time steps
dt = 5e-9  ##s
D_0 = 2.3e-3  ## mm^2/s
dr = np.sqrt(6 * D_0 * dt)
# mm (order of 10^-6)
# about two orders of magnitude lower than the radius of the fibrils : the case in which a particle would pass inside a fibril but end outside
# are therefore not treated since this effect is assumed to be negligible
L_0 = 1e-4  ##mm
T = 0.75  # needs to be correlated to phi (volume fibrils/total volume) via computephi
DT = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# eigenvalues are ordered on the value of the scal prod of eigenvector and k versor
D_1 = []  # principal eigenvalues of DT matrix
D_2 = []  # secondary eigenvalues
D_3 = []  # tertiary eigenvalues

# this is expensivish
# for a simulation, this would need to be calculated about N_p*N_t, going by the paper's numbers
# this would be computed (about 1e5) * (about 1e4) = 1e9 times, which on my machine takes about 1 m


def Fibril(x, y, L_0):
    argx = 2 * np.pi * x / L_0
    argy = 2 * np.pi * y / L_0
    term1 = np.cos(argx)
    term2 = (81 / 320) * ((np.cos(argx)) ** 2)
    term3 = (1 / 15) * ((np.cos(argx)) ** 3)
    term4 = (3 / 320) * ((np.cos(argx)) ** 4)
    term5 = np.cos(argy)
    term6 = (81 / 320) * ((np.cos(argy)) ** 2)
    term7 = (1 / 15) * ((np.cos(argy)) ** 3)
    term8 = (3 / 320) * ((np.cos(argy)) ** 4)
    return term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8


# cannot properly verify numerically,only visually on the plot,seems to work


def ComputePhi(T, points):
    accepted = 0
    for i in range(0, points):
        for j in range(0, points):
            if Fibril(i * 1e-7, j * 1e-7, L_0) > T:
                accepted += 1
    return accepted / (points**2)


# updates the coordinates by uniformily sampling the polar angles (dr is fixed by D_0 and dt)


def Boost(coordinates, dr):
    phi = np.random.uniform(0, np.pi)
    theta = np.random.uniform(0, 2 * np.pi)
    coordinates[0] += dr * np.sin(phi) * np.cos(theta)
    coordinates[1] += dr * np.sin(phi) * np.sin(theta)
    coordinates[2] += dr * np.cos(phi)


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
        np.random.uniform(-1e-4, 1e-4),
        np.random.uniform(-1e-4, 1e-4),
        np.random.uniform(-1e-4, 1e-4),
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
for i in range(0, N_p):
    result = MainLoop(N_t, N_p, dt, dr, L_0, T)
    DT = [[DT[i][j] + result[i][j] for j in range(len(DT[0]))] for i in range(len(DT))]
print(DT)
# %%
