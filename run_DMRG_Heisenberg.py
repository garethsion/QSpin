import numpy as np

from src.DMRG import Position, DMRG



def run_dmrg_heisenberg():
    nsites = 4
    n_states_to_keep = 4
    n_sweeps = 4
    S = DMRG(nsites)

    ###############################################################################
    for iter in range(1, int(nsites / 2)):  # do infinite size dmrg for warmup
        # print("WARMUP ITERATION ", iter, S.dim_l, S.dim_r)
        # Create HL and HR by adding the single sites to the two blocks
        S.build_block_left(iter)
        S.build_block_right(iter)
        # find smallest eigenvalue and eigenvector
        S.ground_state()
        # Calculate density matrix
        S.density_matrix(Position.LEFT)
        # Truncate
        S.truncate(Position.LEFT, n_states_to_keep)
        # Reflect
        S.density_matrix(Position.RIGHT)
        S.truncate(Position.RIGHT, n_states_to_keep)

    first_iter = int(nsites / 2)
    for sweep in range(1, n_sweeps):
        for iter in range(first_iter, nsites - 3):
            # print("LEFT-TO-RIGHT ITERATION ", iter, S.dim_l, S.dim_r)
            # Create HL and HR by adding the single sites to the two blocks
            S.build_block_left(iter)
            S.build_block_right(nsites - iter - 2)
            # find smallest eigenvalue and eigenvector
            S.ground_state()
            # Calculate density matrix
            S.density_matrix(Position.LEFT)
            # Truncate
            S.truncate(Position.LEFT, n_states_to_keep)
        first_iter = 1;
        for iter in range(first_iter, nsites - 3):
            # print("RIGHT-TO-LEFT ITERATION ", iter, S.dim_l, S.dim_r)
            # Create HL and HR by adding the single sites to the two blocks
            S.build_block_right(iter);
            S.build_block_left(nsites - iter - 2)
            # find smallest eigenvalue and eigenvector
            S.ground_state();
            # Calculate density matrix
            S.density_matrix(Position.RIGHT)
            # Truncate
            S.truncate(Position.RIGHT, n_states_to_keep)
    print(S.rho)


if __name__ == '__main__':
    run_dmrg_heisenberg()