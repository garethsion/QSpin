import numpy as np
from scipy.sparse.linalg import eigs
from qutip import *
from qent.dmrg import DMRG

class Entanglement:
    '''
        The Entanglement class provides methods for calculating the bipartite and multipartite entanglement for a given
        state vector or density matrix. This class requires the qutip library.

        Author      : Gareth Siôn Jones
        Date        : Sep. 2022
        Affiliation : University of Oxford
        Department  : Materials
        Version     : 0.1.0
    '''
    def __init__(self, N=3, num_spins=[]):
        self.N = N
        self.num_spins = num_spins
        self.dims = self._hilbert_dims()
        self.ene = []
        self.vev = []
        pass

    def _hilbert_dims(self):
        '''
            Returns the Hilbert space dimensions for a state of a given number of spins.
        :return: list of dimensions
        '''
        return [2**self.num_spins[0], 2**self.num_spins[1], 2**self.num_spins[2]]

    def get_partial_trace(self, rho=None, trace_out_sys='b'):
        '''
            Calculates the partial trace of the given density matrix, with respect to the chosen spin system to
            trace out

        :param rho: density matrix
        :param trace_out_sys: spin system to be traced out of density matrix
        :return: partial trace of density matrix
        '''
        assert rho.shape > (1, 1)
        rho.dims = [[self.dims[0], self.dims[1], self.dims[2] ], [self.dims[0], self.dims[1], self.dims[2] ] ]
        
        if trace_out_sys == 'a':
            return rho.ptrace([1, 2])
        elif trace_out_sys == 'b':
            return rho.ptrace([0, 2])
        elif trace_out_sys == 'c':
            return rho.ptrace([0, 1])

    def tracenorm(self, rho=[]):
        '''
            Calculates the singular value decomposition of a given density matrix

        :param rho: density matrix
        :return: singular value decomposition
        '''
        U, s, V = np.linalg.svd(rho)
        return sum(s)

    def get_partial_transpose(self, rho=None, mask=[1, 0], method="sparse"):
        return partial_transpose(rho, mask, method=method)

    def negativity(self, rho=None):
        return self.tracenorm(rho)-1
    
    def ground_state(self, H=[]):
        Ene, Vec = eigs(H)
        GS = Vec[:,1]
        return Qobj(GS, type='Ket')

    @property
    def energy(self):
        return self.ene, self.vec

    @energy.setter
    def energy(self, method='normal', H=[], n_states_to_keep=100, n_sweeps=6):
        if method == 'normal':
            self.ene, self.vec = eigs(H)
        elif method == 'dmrg':
            S = DMRG(self.N, n_sweeps, n_states_to_keep)
            self.ene, self.vec = S.ground_state()

    def thermal_state(self, method='normal', H=[], T=1, K=1, trace_out_sys='b'):

        if len(self.ene) == 0 or len(self.vec) == 0:
            raise ValueError("The energy setter must be called before running this method")

        if trace_out_sys == 'a':
            dim1 = self.dims[1]
            dim2 = self.dims[2]
        elif trace_out_sys == 'b':
            dim1 = self.dims[0]
            dim2 = self.dims[2]
        elif trace_out_sys == 'c':
            dim1 = self.dims[0]
            dim2 = self.dims[1]

        beta = 1/(K*T)

        Z = np.sum(np.exp(-self.ene * beta))
        prob = (1/Z) * np.exp(-self.ene * beta)

        thermal = np.zeros((dim1*dim2, dim1*dim2), dtype='float')
        thermal = Qobj(thermal, dims=[[dim1, dim2], [dim1, dim2]])

        for num in range(0, len(self.ene)):
            psi = Qobj(self.vec[:, num], type='ket')
            rho = ket2dm(psi)
            rho_tr = self.get_partial_trace(rho=rho, trace_out_sys=trace_out_sys)
            thermal = thermal + prob[num]*rho_tr
        return thermal
