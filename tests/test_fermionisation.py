import unittest
import numpy as np
from src.Fermionisation import JordanWigner


class TestJordanWigner(unittest.TestCase):

    def test_fermionic_operators(self):
        jw = JordanWigner(N=1)
        aj = jw.jordan_wigner_transform(1)
        al = jw.jordan_wigner_transform(2)

        self.assertEqual((aj**2).all(), 0)

        # Anti-Commutation relations
        self.assertEqual((aj*aj.T + aj.T*aj).all(), (np.eye(np.size(aj))).all())

        aj = np.kron(aj, np.eye(2))

        self.assertEqual((aj*al + al*aj).all(), 0)
        self.assertEqual((aj.T*al.T + al.T*aj.T).all(), 0)
        self.assertEqual((aj * al.T + al.T * aj).all(), 0)

if __name__=='__main__':
    unittest.main()