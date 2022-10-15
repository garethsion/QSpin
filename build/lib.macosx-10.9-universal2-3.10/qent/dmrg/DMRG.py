### Modified from A. Feiguin's notebook, found at https://github.com/afeiguin/comp-phys/blob/master/13_01_DMRG.ipynb

#TODO Tidy up the Lanczos algorithm
#TODO Use Numba
#TODO Add more Hamiltonians
#TODO Work out the eigsh() problem with truncate

import numpy as np
import scipy
from scipy.sparse import eye
from qent.spin_systems.Hamiltonians import Hamiltonians
from tqdm import tqdm
from numba import jit
# from src.lanczos import lanczos
from qent.dmrg.c_lanczos import lanczos

class IllegalArgumentError(ValueError):
    pass

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

    def xy_hamiltonian(self, H, J=1, B=0, gamma=0, side='left', size=0, dim=0, splusRL=0, szRL=0):
        if side == 'left':
            H[size] = np.kron(H[size - 1], np.eye(2)) + \
                      0.5*J * np.kron(splusRL[size - 1], self.splus.transpose()) + \
                      0.5*J * np.kron(splusRL[size - 1].transpose(), self.splus) +\
                      0.5*J*gamma( np.kron(splusRL[size - 1], self.splus) +\
                                   np.kron(splusRL[size - 1].transpose(), self.splus.transpose())) -\
                      B * np.kron(self.sz, szRL[size - 1])
        elif side == 'right':
            H[size] = np.kron(np.eye(2), H[size - 1]) + \
                      0.5 * J * np.kron(splusRL[size - 1], self.splus.transpose()) + \
                      0.5 * J * np.kron(splusRL[size - 1].transpose(), self.splus) + \
                      0.5 * J * gamma(np.kron(splusRL[size - 1], self.splus) + \
                                      np.kron(splusRL[size - 1].transpose(), self.splus.transpose())) - \
                      B*np.kron(self.sz, szRL[size - 1])
        return H[size]


class Position:
    LEFT, RIGHT = range(2)


class DMRG(object):

    def __init__(self, _nsites, _nsweeps, _n_states_to_keep):

        self.nsites = _nsites
        self.n_sweeps = _nsweeps
        self.n_states_to_keep = _n_states_to_keep
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

        self.psi = np.zeros((2, 2), dtype=np.float)  # ground state wave function
        self.rho = np.zeros((2, 2), dtype=np.float)  # density matrix

        self.energy = 0
        self.error = 0

    @property
    def hamiltonian(self):
        return self.ham_select

    @hamiltonian.setter
    def hamiltonian(self, model):
        models_list = ['heisenberg', 'xy']
        if not model in models_list:
            raise IllegalArgumentError('The selected Hamiltonian is either not currently supported, or there is a problem in its definition.')
        self.ham_select = model

    # @jit(forceobj=True)
    def build_block_left(self, it):
        self.left_size = it
        self.dim_l = self.HL[self.left_size-1].shape[0]

        # Enlarge left block
        ham = DMRGHamiltonians()

        try:
            ham_select = self.ham_select
        except AttributeError:
            print("The hamiltonian must first be set")

        if ham_select == 'heisenberg':
            self.HL[self.left_size] = ham.heisenberg_interaction(self.HL, side='left', size=self.left_size,
                                                                 dim=self.dim_l, splusRL=self.splusL, szRL=self.szL)
        elif ham_select == 'xy':
            self.HL[self.left_size] = ham.xy_hamiltonian(self.HL, side='left', size=self.left_size,
                                                         dim=self.dim_l, splusRL=self.splusL, szRL=self.szL)
        else:
            raise AttributeError("The Hamiltonian selected is not available")

        self.splusL[self.left_size] = np.kron(np.eye(self.dim_l), self.splus0)
        self.szL[self.left_size] = np.kron(np.eye(self.dim_l), self.sz0)

    # @jit(forceobj=True)
    def build_block_right(self, it):
        self.right_size = it
        self.dim_r = self.HR[self.right_size - 1].shape[0]
        ham = DMRGHamiltonians()

        try:
            ham_select = self.ham_select
        except AttributeError:
            print("The hamiltonian must first be set")

        if self.ham_select == 'heisenberg':
            self.HR[self.right_size] = ham.heisenberg_interaction(self.HR, side='right', size=self.right_size,
                                                                 dim=self.dim_r, splusRL=self.splusR, szRL=self.szR)
        elif self.ham_select == 'xy':
            self.HR[self.right_size] = ham.xy_hamiltonian(self.HR, side='right', size=self.right_size,
                                                                  dim=self.dim_r, splusRL=self.splusR, szRL=self.szR)
        else:
            raise AttributeError("The Hamiltonian selected is not available")

        self.splusR[self.right_size] = np.kron(self.splus0, np.eye(self.dim_r))
        self.szR[self.right_size] = np.kron(self.sz0, np.eye(self.dim_r))

    # @jit(forceobj=True)
    def ground_state(self):
        self.dim_l = self.HL[self.left_size].shape[0]
        self.dim_r = self.HR[self.right_size].shape[0]
        # self.psi.resize((self.dim_l, self.dim_r), refcheck=False)
        self.psi = np.resize(self.psi, (self.dim_l, self.dim_r))
        maxiter = self.dim_l*self.dim_r
        (self.energy, self.psi) = lanczos(self, self.psi, maxiter, 1.e-7)

    # @jit(forceobj=True)
    def density_matrix(self, position):
        if position == Position.LEFT:
            self.rho = np.dot(self.psi, self.psi.transpose())
        else:
            self.rho = np.dot(self.psi.transpose(), self.psi)

    # @jit(forceobj=True)
    def truncate(self, position, m):
        # diagonalize rho
        rho_eig, rho_evec = scipy.sparse.linalg.eigsh(self.rho)
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
            aux.resize((aux.shape[0], m), refcheck=False)
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

    def _infinite_dmrg(self):
        for i in tqdm(range(1, self.n_sweeps)):
            for iter in range(1, int(self.nsites / 2)):  # do infinite size dmrg for warmup
                # print("WARMUP ITERATION ", iter, S.dim_l, S.dim_r)
                # Create HL and HR by adding the single sites to the two blocks
                self.build_block_left(iter)
                self.build_block_right(iter)

                # find smallest eigenvalue and eigenvector
                self.ground_state()

                # Calculate density matrix
                self.density_matrix(Position.LEFT)

                # Truncate
                self.truncate(Position.LEFT, self.n_states_to_keep)

                # Reflect
                self.density_matrix(Position.RIGHT)
                self.truncate(Position.RIGHT, self.n_states_to_keep)

    def build_superblock(self, lsites, rsites):
        self.build_block_left(lsites)
        self.build_block_right(rsites)

    def _finite_dmrg(self):
        first_iter = int(self.nsites / 2)
        for i in tqdm(range(1, self.n_sweeps)):
            for sweep in range(1, self.n_sweeps):
                for i in range(first_iter, self.nsites - 3):
                    # print("LEFT-TO-RIGHT ITERATION ", iter, S.dim_l, S.dim_r)
                    # Create HL and HR by adding the single sites to the two blocks
                    self.build_block_left(i)
                    self.build_block_right(self.nsites - i - 2)
                    # self.build_superblock(i, self.nsites-i-2)
                    # find smallest eigenvalue and eigenvector
                    self.ground_state()
                    # Calculate density matrix
                    self.density_matrix(Position.LEFT)
                    # Truncate
                    self.truncate(Position.LEFT, self.n_states_to_keep)
                first_iter = 1
                for i in range(first_iter, self.nsites - 3):
                    # print("RIGHT-TO-LEFT ITERATION ", iter, S.dim_l, S.dim_r)
                    # Create HL and HR by adding the single sites to the two blocks
                    self.build_block_right(i)
                    self.build_block_left(self.nsites - i - 2)
                    # self.build_superblock(i, self.nsites-i-2)
                    # find smallest eigenvalue and eigenvector
                    self.ground_state()
                    # Calculate density matrix
                    self.density_matrix(Position.RIGHT)
                    # Truncate
                    self.truncate(Position.RIGHT, self.n_states_to_keep)

    # @jit(forceobj=True)
    def get_density(self):
        self._infinite_dmrg()
        self._finite_dmrg()
        return self.rho

    def get_energy(self):
        self._infinite_dmrg()
        self._finite_dmrg()
        return self.energy, self.psi