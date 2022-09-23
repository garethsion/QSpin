import numpy as np

from src.DMRG import Position, DMRG



def run_dmrg_heisenberg():
    nsites = 40
    n_states_to_keep = 10
    n_sweeps = 4
    S = DMRG(nsites, n_sweeps, n_states_to_keep)
    rho = S.get_density()
    print(rho)


if __name__ == '__main__':
    run_dmrg_heisenberg()