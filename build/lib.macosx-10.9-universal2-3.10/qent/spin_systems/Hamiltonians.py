from qent.mbs.Fermionisation import JordanWigner
# from Fermionisation import JordanWigner

import numpy as np
from scipy.sparse import csr_matrix, eye, kron


class Hamiltonians:
    #TODO Make hamiltonians sparse
    def __init__(self, N=1):
        self.N = N
        # self.sx = np.matrix('0 1; 1 0')
        # self.sy = np.matrix('0 -1j; 1j 0')
        # self.sz = np.matrix('1 0; 0 -1')

        self.sx = csr_matrix((np.matrix('0 1; 1 0')), dtype=float)
        self.sy = csr_matrix((np.matrix('0 -1j; 1j 0')), dtype=float)
        self.sz = csr_matrix((np.matrix('1 0; 0 -1')), dtype=float)

        self.sminus = .5 * (self.sx - self.sy)
        self.splus = .5 * (self.sx + self.sy)
        
    def heisenberg_interaction(self, J=1):
        H = csr_matrix((2 ** self.N, 2 ** self.N), dtype=np.float)

        for num in range(0, self.N - 1):
            H = H + (J * kron(eye(2 ** (num)), kron(self.sx, kron(self.sx, eye(2 ** (self.N - num - 2))))) + \
                     J * kron(eye(2 ** (num)), kron(self.sy, kron(self.sy, eye(2 ** (self.N - num - 2))))) + \
                     J * kron(eye(2 ** (num)), kron(self.sz, kron(self.sz, eye(2 ** (self.N - num - 2))))))
        return np.matrix(H.toarray())
    
    def heisenberg_magnetic_field(self, B=1, J=1):
        H = self.heisenberg_interaction(J=J)
        
        for num in range(0, self.N):
            H = H + B * (kron(eye(2**num), kron(self.sz, eye(2**(self.N-num-1)))) )
        return H
    
    def XY_hamiltonian(self, gamma=1, B=1, J=1):
        '''
            Spinless Fermions on an open chain
            
            :param: gamma - anisotropy coefficient 0<=gamma<=1 
            :param: B - magnetic field
            :param: J - exchange interaction
            
            :returns: XY Hamiltonian (2**N X 2**N matrix)
        '''
        
        jw = JordanWigner(self.N)

        op = []
        for i in range(self.N):
            op.append(jw.jordan_wigner_transform(i))
        
        H = 0
        for i in range(self.N - 1):
            H += op[i].T.dot(op[i+1]) - op[i].dot(op[i+1].T)
            H -= gamma*(op[i].T.dot(op[i+1].T) - op[i].dot(op[i+1]))

        for i in range(self.N):
            H -= 2*B*(op[i].dot(op[i].T))
        return H
