import numpy as np
cimport numpy as np
cimport cython
import Fibril

@cython.boundscheck(False)
@cython.wraparound(False)
def GetIntersect(tuple[np.float64_t,np.float64_t,np.float64_t] coordinates, tuple[np.float64_t,np.float64_t,np.float64_t] oldcoord, np.float64_t phi, np.float64_t theta, np.float64_t T, np.float64_t L_0) -> tuple[np.float64_t,np.float64_t,np.float64_t]:
    cdef double x = oldcoord[0]
    cdef double y = oldcoord[1]
    cdef double z = oldcoord[2]
    cdef double[3] checking = [x, y, z]
    cdef double Bordereq = 0.0
    cdef double minBorderDist = Fibril.Fibril(checking[0], checking[1], L_0) - T
    cdef double[3] nearest_point = [x, y, z]
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
                Bordereq = Fibril.Fibril(checking[0], checking[1], L_0) - T
                if Bordereq > 0:
                    break
                if Bordereq > minBorderDist:
                    minBorderDist = Bordereq
                    nearest_point[0] = i
                    nearest_point[1] = j
                i += dx
            nearest_point[2] = z + (
                (nearest_point[0] - oldcoord[0]) / (np.tan(phi) * np.cos(theta))
            )
        else:
            while i >= coordinates[0]:
                j = m * i + q
                checking = [i, j, z]
                Bordereq = Fibril.Fibril(checking[0], checking[1], L_0) - T
                if Bordereq > 0:
                    break
                if Bordereq > minBorderDist:
                    minBorderDist = Bordereq
                    nearest_point[0] = i
                    nearest_point[1] = j
                i += dx
            nearest_point[2] = z + (
                (nearest_point[0] - oldcoord[0]) / (np.tan(phi) * np.cos(theta))
            )
    else:
        j = y + dy
        if oldcoord[1] <= coordinates[1]:  # if y_fin >= y_in
            j = y + dy
            while j <= coordinates[1]:
                checking = [x, j, z]
                Bordereq = Fibril.Fibril(checking[0], checking[1], L_0) - T
                if Bordereq > 0:
                    break
                if Bordereq > minBorderDist:
                    minBorderDist = Bordereq
                    nearest_point[0] = x
                    nearest_point[1] = j
                j += dy
            nearest_point[2] = z + (
                (nearest_point[1] - oldcoord[1]) / (np.tan(phi) * np.sin(theta))
            )
        else:
            j = y + dy
            while j >= coordinates[1]:
                checking = [x, j, z]
                Bordereq = Fibril.Fibril(checking[0], checking[1], L_0) - T
                if Bordereq > 0:
                    break
                if Bordereq > minBorderDist:
                    minBorderDist = Bordereq
                    nearest_point[0] = x
                    nearest_point[1] = j
                j += dy
            nearest_point[2] = z + (
                (nearest_point[1] - oldcoord[1]) / (np.tan(phi) * np.sin(theta))
            )
    return nearest_point
