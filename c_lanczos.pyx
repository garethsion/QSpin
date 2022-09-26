import numpy as np
import cython

from libcpp cimport bool
cimport numpy as np

np.import_array()
DTYPE = np.double

ctypedef np.double_t DTYPE_t

cpdef lanczos(m, seed, int maxiter, double tol, bint use_seed=False, bint force_maxiter=False):
    cdef np.ndarray x1 = seed
    cdef np.ndarray x2 = seed
    cdef np.ndarray gs = seed
    cdef np.ndarray a = np.zeros(100)
    cdef np.ndarray b = np.zeros(100)
    cdef np.ndarray z = np.zeros((100, 100))
    cdef list lvectors = []
    cdef int control_max = maxiter;
    cdef int e0 = 9999
    # cdef bint force_maxiter = force_maxiter

    # if (maxiter == -1):
    #     force_maxiter = False

    if (control_max == 0):
        gs = 1
        maxiter = 1
        return (e0, gs)

    x1[:, :] = 0
    x2[:, :] = 0
    gs[:, :] = 0
    a[:] = 0.0
    b[:] = 0.0
    if (use_seed):
        x1 = seed
    else:
        #x1 = [[ (2 * np.random.random() - 1.) for j in range(x1.shape[1]) ] for i in range(x1.shape[0])]
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                x1[i, j] = (2 * np.random.random() - 1.)

    b[0] = np.sqrt(float(np.tensordot(x1, x1)))
    x1 = x1 / b[0]
    x2[:] = 0
    b[0] = 1.

    e0 = 9999
    nmax = min(99, maxiter)

    for iter in range(1, nmax + 1):
        eini = e0
        if (b[iter - 1] != 0.):
            aux = x1
            x1 = -b[iter - 1] * x2
            x2 = aux / b[iter - 1]

        aux = m.product(x2)

        x1 = x1 + aux
        a[iter] = float(np.tensordot(x1,x2))
        x1 = x1 - x2 * a[iter]

        b[iter] = np.sqrt(float(np.tensordot(x1,x1)))
        lvectors.append(x2)
        z.resize((iter, iter), refcheck=False)
        z[:, :] = 0
        for i in range(0, iter - 1):
            z[i, i + 1] = b[i + 1]
            z[i + 1, i] = b[i + 1]
            z[i, i] = a[i + 1]
        z[iter - 1, iter - 1] = a[iter]
        d, v = np.linalg.eig(z)

        col = 0
        n = 0
        e0 = 9999
        for e in d:
            if (e < e0):
                e0 = e
                col = n
            n += 1
        e0 = d[col]

        if ((force_maxiter and iter >= control_max) or (
                iter >= gs.shape[0] * gs.shape[1] or iter == 99 or abs(b[iter]) < tol) or \
                ((not force_maxiter) and abs(eini - e0) <= tol)):
            # converged
            gs[:, :] = 0.
            for n in range(0, iter):
                gs += v[n, col] * lvectors[n]

            maxiter = iter
            return (e0, gs)  # We return with ground states energy

    return (e0, gs)