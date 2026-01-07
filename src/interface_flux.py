import numpy as np
import params as p
from velocity import wall_shear_stress_endothelium


# NEW: shear-stress dependent permeability xi(sigma) as in the reference
def xi_from_shear(sigma_abs):
    # paper defines a standard wall shear stress wsstand computed from Poiseuille flow
    wsstand = 4.0 * p.nu * p.Ul_max / p.Ly1

    # xi(sigma) = perstand/log(2) * log(1 + 2*wsstand/(|sigma| + wsstand))
    return (p.perstand / np.log(2.0)) * np.log(1.0 + (2.0 * wsstand) / (sigma_abs + wsstand))


def apply_interface_flux(c_LDL_l, c_LDL_i, cl_n, ci_n, dt, hy1, hy2):
    # Interface coupling (Kedem-Katchalsky law)
    sigma = wall_shear_stress_endothelium()
    xi = xi_from_shear(abs(sigma))

    for i in range(1, p.Nx-1):
        flux = xi * (cl_n[0, i] - ci_n[-1, i])

        c_LDL_l[0,i]  -= dt * flux / hy1   # lumen loses LDL
        c_LDL_i[-1,i] += dt * flux / hy2   # intima gains LDL
