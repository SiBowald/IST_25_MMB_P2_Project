# NEW: split from original script into a module

import params as p


def apply_interface_flux(c_LDL_l, c_LDL_i, cl_n, ci_n, dt, hy1, hy2):
    # Interface coupling (Kedem-Katchalsky law)
    for i in range(1, p.Nx-1):
        flux = p.kappa * (cl_n[0, i] - ci_n[-1, i])

        c_LDL_l[0,i]  -= dt * flux / hy1   # lumen loses LDL
        c_LDL_i[-1,i] += dt * flux / hy2   # intima gains LDL
