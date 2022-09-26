import unittest
import numpy as np
from QEnt.spin_systems.Hamiltonians import Hamiltonians


class TestHamiltonians(unittest.TestCase):

    def test_spin_operators(self):
        hams = Hamiltonians(N=3)
        sx = np.matrix(hams.sx.toarray())
        sy = np.matrix(hams.sy.toarray())
        sz = np.matrix(hams.sz.toarray())

        # Commutation relations
        self.assertEqual((sx*sy - sy*sx).all(), (2*1j*sz).all())
        self.assertEqual((sy * sz - sz * sy).all(), (2 * 1j * sx).all())
        self.assertEqual((sz * sx - sx * sz).all(), (2 * 1j * sy).all())
        self.assertEqual((sx * sx - sx * sx).all(), 0)

        # Anti-Commutation relations
        self.assertEqual((sx*sx + sx*sx).all(), (2*np.eye(2)).all())
        self.assertEqual((sx * sy + sy * sx).all(), 0)
        self.assertEqual((sx * sz + sz * sx).all(), 0)
        self.assertEqual((sy * sy + sy * sy).all(), (2*np.eye(2)).all())

    def test_heisenberg_interaction(self):
        hams = Hamiltonians(N=3)
        heis = hams.heisenberg_interaction()

        # Check Hamiltonian is Hermitian
        self.assertEqual(heis.all(), heis.H.all())

    def test_heisenberg_magnetic_field(self):
        hams = Hamiltonians(N=3)
        heis = hams.heisenberg_magnetic_field()

        # Check Hamiltonian is Hermitian
        self.assertEqual(heis.all(), heis.H.all())

    def test_XY_hamiltonian(self):
        hams = Hamiltonians(N=3)
        heis = hams.XY_hamiltonian()

        # Check Hamiltonian is Hermitian
        self.assertEqual(heis.all(), heis.H.all())

if __name__=='__main__':
    unittest.main()