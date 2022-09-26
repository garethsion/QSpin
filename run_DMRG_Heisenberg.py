from QEnt.dmrg.DMRG import DMRG
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

@timeit
def run_dmrg_heisenberg():
    nsites = 4
    n_states_to_keep = 4
    n_sweeps = 6
    S = DMRG(nsites, n_sweeps, n_states_to_keep)
    rho = S.get_density()
    print(rho)


if __name__ == '__main__':
    run_dmrg_heisenberg()