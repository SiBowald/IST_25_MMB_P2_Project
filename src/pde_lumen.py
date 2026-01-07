
import params as p
from velocity import velocity_lumen


def step_lumen_ldl(c_LDL_l, cl_n, dt, y1, hx, hy1):
    # Lumen: LDL transport (FD-BD-CD)
    # PDE:
    # Diffusion: CD, advection: BD, time: FD
    for j in range(1, p.Ny1 - 1):  # j indexes the y-direction

        vx = velocity_lumen(y1[j])  # velocity depends only on y

        for i in range(1, p.Nx - 1):  # i indexes the x-direction
            # Note that x and y are the otherway around here!!!! So that's why j is first and i second!!

            # Diffusion term
            # 2D Laplacian using CD (central differences) in x and y
            laplacian = (cl_n[j, i + 1] - 2 * cl_n[j, i] + cl_n[j, i - 1]) / hx ** 2 + (
                        cl_n[j + 1, i] - 2 * cl_n[j, i] + cl_n[j - 1, i]) / hy1 ** 2

            # Advection term using BD (backward difference), bc vx > 0
            advection = -vx * (cl_n[j, i] - cl_n[j, i - 1]) / hx

            # Time update using FD (forward difference)
            c_LDL_l[j, i] = cl_n[j, i] + dt * (p.Dl * laplacian + advection)

    ### External boundaries (no flux) lumen ###
    # Bottom boundary lumen is the interface with the top boundary intima, so not external!

    # Left boundary: inlet (x=0), using FD
    c_LDL_l[:, 0] = p.Cl_in  # c_LDL_l[:,0] = c_LDL_l[:,1]

    # Right boundary: outlet (x=Lx), using BD
    c_LDL_l[:, -1] = c_LDL_l[:, -2]  # c_LDL_l[:,Nx] = c_LDL_l[:,Nx-1]

    # Top boundary: arterial wall (y=Ly2+Ly1), using BD
    c_LDL_l[-1, :] = c_LDL_l[-2, :]  # c_LDL_l[Ny,:] = c_LDL_l[Ny-1,:]
