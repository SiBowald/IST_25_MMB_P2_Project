# NEW: split from original script into a module

import params as p


def step_intima_ldl_and_reactions(c_LDL_i, c_oxLDL_i, M, F, ci_n, cox_n, M_n, dt, hx, hy2):
    for j in range(1, p.Ny2-1):
        for i in range(1, p.Nx-1):

            laplacian = (ci_n[j,i+1] - 2 * ci_n[j,i] + ci_n[j, i-1]) / hx**2 + (ci_n[j+1, i] - 2 * ci_n[j, i] + ci_n[j-1, i]) / hy2**2

            # NEW: bugfix (missing assignment in original) — update c_LDL_i in place
            c_LDL_i[j, i] = ci_n[j, i] + dt * (p.Di * laplacian - p.r_ox * ci_n[j, i])



    # ODEs for inflammation reactions
    c_oxLDL_i += dt * (p.r_ox * ci_n - p.kF * cox_n * M_n)
    M   += dt * (-p.kF * cox_n * M_n)
    F   += dt * (p.kF * cox_n * M_n)


def apply_intima_boundaries(c_LDL_i):
    ### External boundaries (no flux) intima ###
    ## LDL intima
    # Left boundary (x=0), using FD
    c_LDL_i[:, 0] = c_LDL_i[:, 1] # c_LDL_i[:,0] = c_LDL_i[:,1]

    # Right boundary (x=Lx), using BD
    c_LDL_i[:, -1] = c_LDL_i[:, -2] # c_LDL_i[:,Nx] = c_LDL_i[:,Nx-1]

    # Bottom boundary: media (y=0), using FD
    c_LDL_i[0, :] = c_LDL_i[1, :] # c_LDL_i[0,:] = c_LDL_li[1,:]
