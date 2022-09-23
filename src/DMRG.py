import numpy as np
from src.Hamiltonians import Hamiltonians


class DMRGHamiltonians(Hamiltonians):
    def __init__(self):
        super().__init__()
        self.sz = np.matrix(self.sz.toarray())
        self.splus = np.matrix(self.splus.toarray())

    def heisenberg_interaction(self, H, side='left', size=0, dim=0, splusRL=0, szRL=0):
        if side == 'left':
            H[size] = np.kron(H[size - 1], np.eye(2)) + \
                      np.kron(szRL[size - 1], self.sz) + \
                      0.5 * np.kron(splusRL[size - 1], self.splus.transpose()) + \
                      0.5 * np.kron(splusRL[size - 1].transpose(), self.splus)
        elif side == 'right':
            H[size] = np.kron(np.eye(2), H[size - 1]) + \
                      np.kron(self.sz, szRL[size - 1]) + \
                      0.5 * np.kron(self.splus.transpose(), splusRL[size - 1]) + \
                      0.5 * np.kron(self.splus, splusRL[size - 1].transpose())
        return H[size]


class Position:
    LEFT, RIGHT = range(2)


class DMRG(object):

    def __init__(self, _nsites):

        self.nsites = _nsites
        self.nstates = 2
        self.dim_l = 0  # dimension of left block
        self.dim_r = 0  # dimension of right block
        self.left_size = 0  # number of sites in left block
        self.right_size = 0  # number of sites in right block

        self.sz0 = .5 * np.matrix('-1,0;0,1')
        self.splus0 = np.matrix('0,1;0,0')

        self.HL = []  # left block Hamiltonian
        self.HR = []  # right block Hamiltonian
        self.szL = []  # left block Sz
        self.szR = []  # right block Sz
        self.splusL = []  # left block S+
        self.splusR = []  # right block S+

        for i in range(self.nsites):
            self.HL.append(np.zeros(shape=(2, 2)))
            self.HR.append(np.zeros(shape=(2, 2)))
            self.szL.append(self.sz0)
            self.szR.append(self.sz0)
            self.splusL.append(self.splus0)
            self.splusR.append(self.splus0)

        self.psi = np.zeros(shape=(2, 2))  # ground state wave function
        self.rho = np.zeros(shape=(2, 2))  # density matrix

        self.energy = 0
        self.error = 0

    def build_block_left(self, it):
        self.left_size = it
        self.dim_l = self.HL[self.left_size-1].shape[0]

        # Enlarge left block
        ham = DMRGHamiltonians()
        self.HL[self.left_size] = ham.heisenberg_interaction(self.HL, side='left', size=self.left_size,
                                                             dim=self.dim_l, splusRL=self.splusL, szRL=self.szL)

        self.splusL[self.left_size] = np.kron(np.eye(self.dim_l), self.splus0)
        self.szL[self.left_size] = np.kron(np.eye(self.dim_l), self.sz0)

    def build_block_right(self, it):
        self.right_size = it
        self.dim_r = self.HR[self.right_size - 1].shape[0]
        ham = DMRGHamiltonians()
        self.HR[self.right_size] = ham.heisenberg_interaction(self.HR, side='right', size=self.right_size,
                                                             dim=self.dim_r, splusRL=self.splusR, szRL=self.szR)
        self.splusR[self.right_size] = np.kron(self.splus0, np.eye(self.dim_r))
        self.szR[self.right_size] = np.kron(self.sz0, np.eye(self.dim_r))

    def ground_state(self):
        self.dim_l = self.HL[self.left_size].shape[0]
        self.dim_r = self.HR[self.right_size].shape[0]
        self.psi.resize((self.dim_l, self.dim_r))
        # maxit = self.dim_l*self.dim_r
        # (self.energy, self.psi) = lanczos(self.psi, maxit, 1e-7)
        (self.energy, self.psi) = np.linalg.eigh(self.psi)

    def density_matrix(self, position):
        if position == Position.LEFT:
            self.rho = np.dot(self.psi, self.psi.transpose())
        else:
            self.rho = np.dot(self.psi.transpose(), self.psi)

    def truncate(self, position, m):
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