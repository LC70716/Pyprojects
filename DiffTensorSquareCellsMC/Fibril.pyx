import numpy as np
cimport numpy as np

def Fibril(np.float64_t x, np.float64_t y, np.float64_t L_0) -> np.float64_t:
    cdef np.float64_t argx = 2 * np.pi * x / L_0
    cdef np.float64_t argy = 2 * np.pi * y / L_0
    cdef np.float64_t term1 = np.cos(argx)
    cdef np.float64_t term2 = (81 / 320) * (term1**2)
    cdef np.float64_t term3 = (1 / 15) * (term1**3)
    cdef np.float64_t term4 = (3 / 320) * (term1**4)
    cdef np.float64_t term5 = np.cos(argy)
    cdef np.float64_t term6 = (81 / 320) * (term5**2)
    cdef np.float64_t term7 = (1 / 15) * (term5**3)
    cdef np.float64_t term8 = (3 / 320) * (term5**4)
    return term1 - term2 + term3 - term4 + term5 - term6 + term7 - term8