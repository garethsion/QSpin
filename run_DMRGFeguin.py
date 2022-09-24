from src.DMRGFeiguin import DMRGSystem, Position


def run():
    nsites = 10
    n_states_to_keep = 10
    n_sweeps = 4
    S = DMRGSystem(nsites)
    ###############################################################################
    for iter in range(1, int(nsites / 2)):  # do infinite size dmrg for warmup
        # print("WARMUP ITERATION ", iter, S.dim_l, S.dim_r)
        # Create HL and HR by adding the single sites to the two blocks
        S.BuildBlockLeft(iter)
        S.BuildBlockRight(iter)
        # find smallest eigenvalue and eigenvector
        S.GroundState()
        # Calculate density matrix
        S.DensityMatrix(Position.LEFT)
        # Truncate
        S.Truncate(Position.LEFT, n_states_to_keep)
        # Reflect
        S.DensityMatrix(Position.RIGHT)
        S.Truncate(Position.RIGHT, n_states_to_keep)

    first_iter = int(nsites / 2)
    for sweep in range(1, n_sweeps):
        for iter in range(first_iter, nsites - 3):
            # print("LEFT-TO-RIGHT ITERATION ", iter, S.dim_l, S.dim_r)
            # Create HL and HR by adding the single sites to the two blocks
            S.BuildBlockLeft(iter)
            S.BuildBlockRight(nsites - iter - 2)
            # find smallest eigenvalue and eigenvector
            S.GroundState()
            # Calculate density matrix
            S.DensityMatrix(Position.LEFT)
            # Truncate
            S.Truncate(Position.LEFT, n_states_to_keep)
        first_iter = 1;
        for iter in range(first_iter, nsites - 3):
            # print("RIGHT-TO-LEFT ITERATION ", iter, S.dim_l, S.dim_r)
            # Create HL and HR by adding the single sites to the two blocks
            S.BuildBlockRight(iter);
            S.BuildBlockLeft(nsites - iter - 2)
            # find smallest eigenvalue and eigenvector
            S.GroundState();
            # Calculate density matrix
            S.DensityMatrix(Position.RIGHT)
            # Truncate
            S.Truncate(Position.RIGHT, n_states_to_keep)
    print(S.rho)

if __name__=='__main__':
    run()