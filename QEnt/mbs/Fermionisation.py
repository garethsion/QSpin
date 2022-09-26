import numpy as np
class Fermionisation:
    def __init__(self, N=1):
        self.N = N
        self.sx = np.matrix('0 1; 1 0')
        self.sy = np.matrix('0 -1j; 1j 0')
        self.sz = np.matrix('1 0; 0 -1')
        self.splus = np.real((self.sx + 1j*self.sy)/2)
        self.sminus = np.real((self.sx - 1j*self.sy)/2)

class JordanWigner(Fermionisation):
    def __init__(self, N=1):
        Fermionisation.__init__(self, N=N)
    
    def nested_kronecker_product(self, op=[]):
        '''
            Creates a nested Kronecker product for Pauli operators
        :param op: Operators, usually Pauli
        :return: kronecker product of operators
        '''
        if len(op) == 2:
            return np.kron(op[0], op[1])
        else:
            return np.kron(op[0], self.nested_kronecker_product(op[1:]))
        
    def jordan_wigner_transform(self, j):
        '''
            Calculates the Jordan-Wigner transformation of Pauli operators.
        :param j: site number
        :return: matrix of operators
        '''
        I = np.eye(2)
        
        operators = []
        for k in range(j):
            operators.append(self.sz)
        operators.append(self.splus)
        
        for k in range(self.N-j-1):
            operators.append(I)
        return -self.nested_kronecker_product(operators)
        
    def fourier_transformation(self):
        operators = []
        for m in range(self.N):
            operators.append((1/np.sqrt(2)) * np.exp(1j * m * ((2*np.pi*m)/self.N)) * self.jordan_wigner_transform(m))
        return operators
