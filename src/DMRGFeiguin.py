def psi_dot_psi(psi1, psi2):
    x = 0.
    for i in range(psi1.shape[0]):
        for j in range(psi2.shape[1]):
            x += psi1[i, j] * psi2[i, j]
    return x


def lanczos(m, seed, maxiter, tol, use_seed=False, force_maxiter=False):
    x1 = seed
    x2 = seed
    gs = seed
    a = np.zeros(100)
    b = np.zeros(100)
    z = np.zeros((100, 100))
    lvectors = []
    control_max = maxiter;
    e0 = 9999

    if (maxiter == -1):
        force_maxiter = False

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
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                x1[i, j] = (2 * np.random.random() - 1.)

    #    x1[:,:] = 1
    b[0] = psi_dot_psi(x1, x1)
    b[0] = np.sqrt(b[0])
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
        a[iter] = psi_dot_psi(x1, x2)
        x1 = x1 - x2 * a[iter]

        b[iter] = psi_dot_psi(x1, x1)
        b[iter] = np.sqrt(b[iter])
        lvectors.append(x2)
        #        print "Iter =",iter,a[iter],b[iter]
        z.resize((iter, iter))
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

        # print("Iter = ",iter," Ener = ",e0)
        if ((force_maxiter and iter >= control_max) or (
                iter >= gs.shape[0] * gs.shape[1] or iter == 99 or abs(b[iter]) < tol) or \
                ((not force_maxiter) and abs(eini - e0) <= tol)):
            # converged
            gs[:, :] = 0.
            for n in range(0, iter):
                gs += v[n, col] * lvectors[n]

            # print("E0 = ", e0)
            maxiter = iter
            return (e0, gs)  # We return with ground states energy

    return (e0, gs)


import numpy as np


class Position:
    LEFT, RIGHT = range(2)


class DMRGSystem(object):

    def __init__(self, _nsites):

        # Single site operators
        self.nsites = _nsites
        self.nstates = 2
        self.dim_l = 0  # dimension of the left block
        self.dim_r = 0  # dimension of the right block
        self.left_size = 0  # number of sites in the left block
        self.right_size = 0  # number of sites in the right block

        self.sz0 = np.zeros(shape=(2, 2))  # single site Sz
        self.splus0 = np.zeros(shape=(2, 2))  # single site S+
        self.sz0[0, 0] = -0.5
        self.sz0[1, 1] = 0.5
        self.splus0[1, 0] = 1.0

        # Useful structures to store the matrices

        self.HL = []  # left block Hamiltonian
        self.HR = []  # right block Hamiltonian
        self.szL = []  # left block Sz
        self.szR = []  # right block Sz
        self.splusL = []  # left block S+
        self.splusR = []  # right block S+

        zero_matrix = np.zeros(shape=(2, 2))
        for i in range(self.nsites):
            self.HL.append(zero_matrix)
            self.HR.append(zero_matrix)
            self.szL.append(self.sz0)
            self.szR.append(self.sz0)
            self.splusL.append(self.splus0)
            self.splusR.append(self.splus0)

        self.psi = np.zeros(shape=(2, 2))  # g.s. wave function
        self.rho = np.zeros(shape=(2, 2))  # density matrix

        self.energy = 0.
        self.error = 0.

    #######################################

    def BuildBlockLeft(self, iter):
        self.left_size = iter
        self.dim_l = self.HL[self.left_size - 1].shape[0]
        I_left = np.eye(self.dim_l)
        I2 = np.eye(2)
        # enlarge left block:
        self.HL[self.left_size] = np.kron(self.HL[self.left_size - 1], I2) + \
                                  np.kron(self.szL[self.left_size - 1], self.sz0) + \
                                  0.5 * np.kron(self.splusL[self.left_size - 1], self.splus0.transpose()) + \
                                  0.5 * np.kron(self.splusL[self.left_size - 1].transpose(), self.splus0)
        self.splusL[self.left_size] = np.kron(I_left, self.splus0)
        self.szL[self.left_size] = np.kron(I_left, self.sz0)

    def BuildBlockRight(self, iter):
        self.right_size = iter
        self.dim_r = self.HR[self.right_size - 1].shape[0]
        I_right = np.eye(self.dim_r)
        I2 = np.eye(2)
        # enlarge right block:
        self.HR[self.right_size] = np.kron(I2, self.HR[self.right_size - 1]) + \
                                   np.kron(self.sz0, self.szR[self.right_size - 1]) + \
                                   0.5 * np.kron(self.splus0.transpose(), self.splusR[self.right_size - 1]) + \
                                   0.5 * np.kron(self.splus0, self.splusR[self.right_size - 1].transpose())
        self.splusR[self.right_size] = np.kron(self.splus0, I_right)
        self.szR[self.right_size] = np.kron(self.sz0, I_right)

    def GroundState(self):
        self.dim_l = self.HL[self.left_size].shape[0]
        self.dim_r = self.HR[self.right_size].shape[0]
        self.psi.resize((self.dim_l, self.dim_r))
        maxiter = self.dim_l * self.dim_r

        (self.energy, self.psi) = lanczos(self, self.psi, maxiter, 1.e-7)
        vec, ene = np.linalg.eigh(self.psi)

    def DensityMatrix(self, position):
        # Calculate density matrix
        if (position == Position.LEFT):
            self.rho = np.dot(self.psi, self.psi.transpose())
        else:
            self.rho = np.dot(self.psi.transpose(), self.psi)

    def Truncate(self, position, m):
        # diagonalize rho
        rho_eig, rho_evec = np.linalg.eigh(self.rho)
        self.nstates = m
        rho_evec = np.real(rho_evec)
        rho_eig = np.real(rho_eig)

        # calculate the truncation error for a given number of states m
        # Reorder eigenvectors and trucate
        index = np.argsort(rho_eig)
        # for e in index:
        #     print("RHO EIGENVALUE ", rho_eig[e])
        error = 0.
        if (m < rho_eig.shape[0]):
            for i in range(index.shape[0] - m):
                error += rho_eig[index[i]]
        # print("Truncation error = ", error)

        aux = np.copy(rho_evec)
        if (self.rho.shape[0] > m):
            aux.resize((aux.shape[0], m))
            n = 0
            for i in range(index.shape[0] - 1, index.shape[0] - 1 - m, -1):
                aux[:, n] = rho_evec[:, index[i]]
                n += 1
        rho_evec = aux

        #        rho_evec = np.eye(self.rho.shape[0])

        # perform transformation:
        U = rho_evec.transpose()
        if (position == Position.LEFT):
            aux2 = np.dot(self.HL[self.left_size], rho_evec)
            self.HL[self.left_size] = np.dot(U, aux2)
            aux2 = np.dot(self.splusL[self.left_size], rho_evec)
            self.splusL[self.left_size] = np.dot(U, aux2)
            aux2 = np.dot(self.szL[self.left_size], rho_evec)
            self.szL[self.left_size] = np.dot(U, aux2)
        else:
            aux2 = np.dot(self.HR[self.right_size], rho_evec)
            self.HR[self.right_size] = np.dot(U, aux2)
            aux2 = np.dot(self.splusR[self.right_size], rho_evec)
            self.splusR[self.right_size] = np.dot(U, aux2)
            aux2 = np.dot(self.szR[self.right_size], rho_evec)
            self.szR[self.right_size] = np.dot(U, aux2)

    def product(self, psi):
        npsi = np.dot(self.HL[self.left_size], psi)
        npsi += np.dot(psi, self.HR[self.right_size].transpose())

        # Sz.Sz
        tmat = np.dot(psi, self.szR[self.right_size].transpose())
        npsi += np.dot(self.szL[self.left_size], tmat)
        # S+.S-
        tmat = np.dot(psi, self.splusR[self.right_size]) * 0.5
        npsi += np.dot(self.splusL[self.left_size], tmat)
        # S-.S+
        tmat = np.dot(psi, self.splusR[self.right_size].transpose()) * 0.5
        npsi += np.dot(self.splusL[self.left_size].transpose(), tmat)

        return npsi