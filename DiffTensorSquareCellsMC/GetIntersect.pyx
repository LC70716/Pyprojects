import numpy as np
cimport numpy as np
cimport cython
import Fibril

@cython.boundscheck(False)
@cython.wraparound(False)
def GetIntersect(tuple[np.float64_t,np.float64_t,np.float64_t] coordinates, tuple[np.float64_t,np.float64_t,np.float64_t] oldcoord, np.float64_t phi, np.float64_t theta, np.float64_t T, np.float64_t L_0, double tolerance=1e-8) -> tuple[np.float64_t,np.float64_t,np.float64_t]:

    def find_midpoint(a, b):
        return [(a[i] + b[i]) / 2 for i in range(3)] 
 
    a = oldcoord
    b = coordinates
    diff = [b[i]-a[i] for i in range(3)]
    while np.linalg.norm(diff) > tolerance:  # Check distance instead of steps
        mid = find_midpoint(a, b)
        f_mid = Fibril.Fibril(mid[0], mid[1], L_0) - T

        if f_mid == 0: 
            return mid  # Exact zero found!

        if np.sign(f_mid) == np.sign(Fibril.Fibril(a[0], a[1], L_0) - T):
            a = mid   # Zero is in the second half
        else:
            b = mid   # Zero is in the first half
        diff = [b[i]-a[i] for i in range(3)]
    # Return the approximate zero-crossing point (midpoint at termination)
    z_coord = oldcoord[2] + ((mid[0] - oldcoord[0]) / (np.tan(phi) * np.cos(theta)))  # Or alternative calculation
    return [mid[0], mid[1], z_coord] 