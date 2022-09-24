import time
from src.DMRG import Position, DMRG



def run_dmrg_heisenberg():
    nsites = 10
    n_states_to_keep = 10
    n_sweeps = 6
    S = DMRG(nsites, n_sweeps, n_states_to_keep)
    rho = S.get_density()
    print(rho)


if __name__ == '__main__':
    tstrt = time.time()
    run_dmrg_heisenberg()
    tstp = time.time()

    time_taken = tstp-tstrt
    print(time_taken)