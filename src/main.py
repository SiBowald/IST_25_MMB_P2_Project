#!/usr/bin/env python
# coding: utf-8

# ## Atherosclerosis Model

# In[93]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt

import params as p
from grid import build_grid
from pde_lumen import step_lumen_ldl
from pde_intima import step_intima_ldl_and_reactions, apply_intima_boundaries
from interface_flux import apply_interface_flux
from plotting import maybe_plot


# In[96]:


# NEW: grid moved to grid.py
x, y1, y2, hx, hy1, hy2 = build_grid()


# In[98]:


### Define arrays for fields ###

# Lumen
c_LDL_l = np.zeros((p.Ny1, p.Nx))     # LDL concentration lumen

# Intima
c_LDL_i = np.zeros((p.Ny2, p.Nx))     # LDL concentration intima
c_oxLDL_i =np.zeros((p.Ny2, p.Nx))     # ox-LDL concentration intima
M = np.zeros((p.Ny2, p.Nx))           # macrophages (conc?????)
F = np.zeros((p.Ny2, p.Nx))            # foam cells (conc?????)


# In[102]:


### Initial conditions ###
c_LDL_l[:, :] = p.Cl_in             # initial LDL in lumen

# no cells or LDL in intima at t=0
c_LDL_i[:, :] = 0.0
c_oxLDL_i[:, :] = 0.0
M[:, :] = 0.0
F[:, :] = 0.0


# In[104]:


### Explicit stability conditions time step ###

dt_adv = hx / p.Ul_max
dt_diff_l = 1.0 / (2 * p.Dl * (1/hx**2 + 1/hy1**2))
dt_diff_i = 1.0 / (2 * p.Di * (1/hx**2 + 1/hy2**2))
dt_react  = 1.0 / max(p.r_ox, p.kF * p.Cl_in)

dt = 0.5 * min(dt_adv, dt_diff_l, dt_diff_i, dt_react)

T  = 0.1
Nt = int(T / dt)

print(f"dt = {dt:.2e}, Nt = {Nt}")


# In[109]:


### Time loop ###
for n in range(Nt):       # n indexes the t

    # Copy so derivatives are computed from the old time level
    cl_n = c_LDL_l.copy()      # LDL lumen
    ci_n = c_LDL_i.copy()      # LDL intima
    cox_n = c_oxLDL_i.copy()         # ox-LDL intima
    M_n = M.copy()             # macrophages intima
    F_n = F.copy()             # foam cells intima

    # NEW: lumen step moved to pde_lumen.py
    step_lumen_ldl(c_LDL_l, cl_n, dt, y1, hx, hy1)

    # NEW: intima step moved to pde_intima.py
    step_intima_ldl_and_reactions(c_LDL_i, c_oxLDL_i, M, F, ci_n, cox_n, M_n, dt, hx, hy2)

    # NEW: interface coupling moved to interface_flux.py
    apply_interface_flux(c_LDL_l, c_LDL_i, cl_n, ci_n, dt, hy1, hy2)

    # NEW: intima boundaries moved to pde_intima.py
    apply_intima_boundaries(c_LDL_i)

    # NEW: plotting moved to plotting.py
    maybe_plot(n, dt, x, y1, y2, c_LDL_i, c_LDL_l)

plt.show()
