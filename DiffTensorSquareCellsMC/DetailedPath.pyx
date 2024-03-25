import numpy as np
cimport numpy as np
cimport cython
import Fibril

@cython.boundscheck(False)
@cython.wraparound(False)
def DetailedPath(tuple[np.float64_t,np.float64_t,np.float64_t] coordinates, tuple[np.float64_t,np.float64_t,np.float64_t] oldcoord, np.float64_t T, np.float64_t L_0) -> bool :
    cdef double x = oldcoord[0]
    cdef double y = oldcoord[1]
    cdef double z = oldcoord[2]
    cdef double[3] checking = [x, y, z]
    cdef double dx = (coordinates[0] - oldcoord[0]) / 50.0
    cdef double dy = (coordinates[1] - oldcoord[1]) / 50.0
    cdef double m = 0.0
    cdef double q = 0.0
    cdef double i = 0
    cdef double j = 0

    if dx != 0.0:
        i = x + dx
        m = (coordinates[1] - oldcoord[1]) / (coordinates[0] - oldcoord[0])
        q = coordinates[1] - m * coordinates[0]
        if oldcoord[0] <= coordinates[0]: 
            while i <= coordinates[0]:
                j = m * i + q
                checking = [i, j, z]
                if Fibril.Fibril(checking[0], checking[1], L_0) - T > 0:
                    return True  # Early termination
                i += dx
        else:
            while i >= coordinates[0]:
                j = m * i + q
                checking = [i, j, z]
                if Fibril.Fibril(checking[0], checking[1], L_0) - T > 0:
                    return True  # Early termination
                i += dx
    else:
        j = y + dy
        if oldcoord[1] <= coordinates[1]:
            while j <= coordinates[1]:
                checking = [x, j, z]
                if Fibril.Fibril(checking[0], checking[1], L_0) - T > 0:
                    return True  # Early termination
                j += dy
        else:
            while j >= coordinates[1]:
                checking = [x, j, z]
                if Fibril.Fibril(checking[0], checking[1], L_0) - T > 0:
                    return True  # Early termination
                j += dy

    return False 
