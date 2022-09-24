### Modified from A. Feiguin's notebook, found at https://github.com/afeiguin/comp-phys/blob/master/13_01_DMRG.ipynb

#TODO Tidy up the Lanczos algorithm
#TODO Use Numba
#TODO Add more Hamiltonians
#TODO Work out the eigsh() problem with truncate

import numpy as np
import scipy
from scipy.sparse import eye, kron, csr_matrix
from src.Hamiltonians import Hamiltonians
from tqdm import tqdm
from numba import jit

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


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

class DMRGHamiltonians(Hamiltonians):
    def __init__(self):
        super().__init__()
        # self.sz = np.matrix(self.sz.toarray())
        # self.splus = np.matrix(self.splus.toarray())

    def heisenberg_interaction(self, H, side='left', size=0, dim=0, splusRL=0, szRL=0):
        if side == 'left':
            # H[size] = np.kron(H[size - 1], np.eye(2)) + \
            #           np.kron(szRL[size - 1], self.sz) + \
            #           0.5 * np.kron(splusRL[size - 1], self.splus.transpose()) + \
            #           0.5 * np.kron(splusRL[size - 1].transpose(), self.splus)
            H[size] = kron(H[size - 1], eye(2)) + \
                      kron(szRL[size - 1], self.sz) + \
                      0.5 * kron(splusRL[size - 1], self.splus.transpose()) + \
                      0.5 * kron(splusRL[size - 1].transpose(), self.splus)
        elif side == 'right':
            # H[size] = np.kron(np.eye(2), H[size - 1]) + \
            #           np.kron(self.sz, szRL[size - 1]) + \
            #           0.5 * np.kron(self.splus.transpose(), splusRL[size - 1]) + \
            #           0.5 * np.kron(self.splus, splusRL[size - 1].transpose())
            H[size] = kron(eye(2), H[size - 1]) + \
                      kron(self.sz, szRL[size - 1]) + \
                      0.5 * kron(self.splus.transpose(), splusRL[size - 1]) + \
                      0.5 * kron(self.splus, splusRL[size - 1].transpose())
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

        # self.sz0 = .5 * np.matrix('-1,0;0,1')
        # self.splus0 = np.matrix('0,1;0,0')

        self.sx0 = csr_matrix((np.matrix('0 1; 1 0')), dtype=float)
        self.sy0 = csr_matrix((np.matrix('0 -1j; 1j 0')), dtype=float)
        self.sz0 = csr_matrix((np.matrix('1 0; 0 -1')), dtype=float)

        self.sminus0 = .5 * (self.sx0 - self.sy0)
        self.splus0 = .5 * (self.sx0 + self.sy0)

        self.HL = []  # left block Hamiltonian
        self.HR = []  # right block Hamiltonian
        self.szL = []  # left block Sz
        self.szR = []  # right block Sz
        self.splusL = []  # left block S+
        self.splusR = []  # right block S+

        for i in range(self.nsites):
            # self.HL.append(np.zeros(shape=(2, 2)))
            # self.HR.append(np.zeros(shape=(2, 2)))
            self.HL.append(csr_matrix((2, 2), dtype=np.float))
            self.HR.append(csr_matrix((2, 2), dtype=np.float))
            self.szL.append(self.sz0)
            self.szR.append(self.sz0)
            self.splusL.append(self.splus0)
            self.splusR.append(self.splus0)

        self.psi = csr_matrix((2, 2), dtype=np.float)  # ground state wave function
        self.rho = csr_matrix((2, 2), dtype=np.float)  # density matrix

        self.energy = 0
        self.error = 0

    def build_block_left(self, it):
        self.left_size = it
        self.dim_l = self.HL[self.left_size-1].shape[0]

        # Enlarge left block
        ham = DMRGHamiltonians()
        # print("Appending to HL index {:d}".format(self.left_size))
        self.HL[self.left_size] = ham.heisenberg_interaction(self.HL, side='left', size=self.left_size,
                                                             dim=self.dim_l, splusRL=self.splusL, szRL=self.szL)

        # self.splusL[self.left_size] = np.kron(np.eye(self.dim_l), self.splus0)
        # self.szL[self.left_size] = np.kron(np.eye(self.dim_l), self.sz0)
        self.splusL[self.left_size] = kron(eye(self.dim_l), self.splus0)
        self.szL[self.left_size] = kron(eye(self.dim_l), self.sz0)

    def build_block_right(self, it):
        self.right_size = it
        self.dim_r = self.HR[self.right_size - 1].shape[0]
        ham = DMRGHamiltonians()
        self.HR[self.right_size] = ham.heisenberg_interaction(self.HR, side='right', size=self.right_size,
                                                             dim=self.dim_r, splusRL=self.splusR, szRL=self.szR)
        # self.splusR[self.right_size] = np.kron(self.splus0, np.eye(self.dim_r))
        # self.szR[self.right_size] = np.kron(self.sz0, np.eye(self.dim_r))
        self.splusR[self.right_size] = kron(self.splus0, eye(self.dim_r))
        self.szR[self.right_size] = kron(self.sz0, eye(self.dim_r))

    @timeit
    def ground_state(self):
        self.dim_l = self.HL[self.left_size].shape[0]
        self.dim_r = self.HR[self.right_size].shape[0]
        self.psi.resize((self.dim_l, self.dim_r))
        maxiter = self.dim_l*self.dim_r
        (self.energy, self.psi) = lanczos(self, self.psi, maxiter, 1.e-7)
        # (self.energy, self.psi) = np.linalg.eigh(self.psi)

    def density_matrix(self, position):
        if position == Position.LEFT:
            self.rho = self.psi.dot(self.psi.transpose())
        else:
            self.rho = self.psi.transpose().dot(self.psi)

    def truncate(self, position, m):
        # diagonalize rho
        rho_eig, rho_evec = scipy.sparse.linalg.eigsh(self.rho.toarray())
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
            aux2 = self.HL[self.left_size].dot(rho_evec)
            self.HL[self.left_size] = np.dot(U, aux2)
            aux2 = self.splusL[self.left_size].dot(rho_evec)
            self.splusL[self.left_size] = np.dot(U, aux2)
            aux2 = self.szL[self.left_size].dot(rho_evec)
            self.szL[self.left_size] = np.dot(U, aux2)
        else:
            aux2 = self.HR[self.right_size].dot(rho_evec)
            self.HR[self.right_size] = np.dot(U, aux2)
            aux2 = self.splusR[self.right_size].dot(rho_evec)
            self.splusR[self.right_size] = np.dot(U, aux2)
            aux2 = self.szR[self.right_size].dot(rho_evec)
            self.szR[self.right_size] = np.dot(U, aux2)

    def product(self, psi):
        npsi = self.HL[self.left_size].dot(psi)
        npsi += psi.dot(self.HR[self.right_size].transpose())

        # Sz.Sz
        tmat = psi.dot(self.szR[self.right_size].transpose())
        npsi += self.szL[self.left_size].dot(tmat)
        # S+.S-
        tmat = psi.dot(self.splusR[self.right_size]) * 0.5
        npsi += self.splusL[self.left_size].dot(tmat)
        # S-.S+
        tmat = psi.dot(self.splusR[self.right_size].transpose()) * 0.5
        npsi += self.splusL[self.left_size].transpose().dot(tmat)

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

    def _finite_dmrg(self):
        first_iter = int(self.nsites / 2)
        for i in tqdm(range(1, self.n_sweeps)):
            for sweep in range(1, self.n_sweeps):
                for i in range(first_iter, self.nsites - 3):
                    # print("LEFT-TO-RIGHT ITERATION ", iter, S.dim_l, S.dim_r)
                    # Create HL and HR by adding the single sites to the two blocks
                    self.build_block_left(i)
                    self.build_block_right(self.nsites - i - 2)
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