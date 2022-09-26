import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from QEnt.spin_systems.Hamiltonians import Hamiltonians
from QEnt.mbs.Entanglement import Entanglement

mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.frameon'] = True
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.frameon'] = False


def run_thermal_state_sweep():
    gamma = 0
    J = 1
    B = 0
    N = 3

    hams = Hamiltonians(N=N)
    H = hams.XY_hamiltonian(gamma=gamma, J=J, B=B)

    ent = Entanglement(N=N, num_spins=[1, 1, 1])
    tvec = np.arange(0.1, 2.1, 0.1)

    negativity = []
    for t in tvec:
        rho_th = ent.thermal_state(H=H, T=t, K=1, trace_out_sys='b')
        pac_ta = ent.get_partial_transpose(rho=rho_th, mask=[1, 0], method="sparse")
        negativity.append(ent.negativity(pac_ta))

    fig = plt.figure()
    plt.plot(tvec, negativity, 's-', color='k')
    plt.xlabel('T/J')
    plt.ylabel('Negativity')
    # plt.title('Single Spins at Both Ends')
    plt.show()


if __name__ == '__main__':
    run_thermal_state_sweep()