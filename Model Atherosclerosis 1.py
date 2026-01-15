#!/usr/bin/env python
# coding: utf-8

# ## Atherosclerosis Model

# In[93]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt


# #### Parameters

# In[95]:


### Parameters ###

# Lumen
rho = 1.05       # [g/cm^3] blood density
nu = 0.035       # [g/(cm s) dynamic viscosity
Dl = 2.867e-7    # [cm^s/s] LDL diffusion coefficient
Ul_max = 24.0    # [cm/s] maximum axial velocity
p_out = 870.0    # [mmHg] outlet pressure
Cl_in = 3.12e-6  # [g/cm^3] inlet LDL concentration

# Intima
kappa = 8.7e-13      # [cm^2] permeability CHECKKKKK
mu = 0.03655     # [g/(cm s)] viscosity
Di = 1.20e-7     # [cm^2/s] LDL diffusion in intima
p_med = 800.0    # [mmHg] medial pressure

# Inflammation
d_ox = 1.0e-3    # [cm^2/s] ox-LDL diffusion
dM = 1.0e-5      # [cm^2/s] macrophage diffusion
dS = 1.0e-3      # [cm^2/s] cytokine diffusion
kF = 1.0         # [cm^3/(g s)] foam cell formation rate
r_ox = 0.5       # [1/s] LDL oxidation rate
labda = 10.0     # [1/s] cytokine degradation
gamma = 1.0      # [1/s] cytokine production

cox_th = 0.01




# In[96]:


### Numerical grid parameters ###

# Domain artery
Lx = 5.0         # [cm] length of the artery segment
Ly2 = 0.02       # [cm] height of the intima (IDK why this, should find reference)
Ly1 = 0.5        # [cm] height of the lumen 

# Number of grid points
Nx = 50          # axial direction (took it from example teacher, but maybe find reference?)
Ny1 = 25         # lumen (same as above)
Ny2 = 25         # intima (same as above)

# Spatial discretization 
hx = Lx / (Nx - 1)
hy1 = Ly1 / (Ny1 -1)
hy2 = Ly2 / (Ny2 -1)

# Arrays
x = np.linspace(0.0, Lx, Nx)
y2 = np.linspace(0.0, Ly2, Ny2)        # intima (lower domain (2))
y1 = np.linspace(Ly2, Ly2 + Ly1, Ny1)  # lumen  (upper domain (1))


# In[98]:


### Define arrays for fields ###

# Lumen
c_LDL_l = np.zeros((Ny1, Nx))     # LDL concentration lumen

# Intima
c_LDL_i = np.zeros((Ny2, Nx))     # LDL concentration intima
c_oxLDL_i =np.zeros((Ny2, Nx))     # ox-LDL concentration intima
M = np.zeros((Ny2, Nx))           # macrophages (conc?????)
F = np.zeros((Ny2, Nx))            # foam cells (conc?????)
S = np.zeros((Ny2, Nx))            #

# Wall Shear Stress + Permeability (single Gaussian bump in x)
perstand = 1.07e-11
wsstand  = 4.0 * nu * Ul_max / Ly1   # baseline/reference WSS

# --- Gaussian bump parameters ---
x0 = 0.55 * Lx      # center of the bump [cm]
w  = 0.25           # width (std dev) [cm]
peak_factor = 20.55  # +2055% peak (so peak ~ 21.55*baseline) //0.55 -> +55% peak (so peak ~ 1.55*baseline)

# local max velocity profile along x
bump = 1.0 + peak_factor * np.exp(-0.5 * ((x - x0) / w) ** 2)
Ul_max_x = Ul_max * bump

# local wall shear stress tau_w(x) from Poiseuille in a flat channel
du_dy_x = 4.0 * Ul_max_x / Ly1
tau_w_x = nu * np.abs(du_dy_x)

# formula permeability in paper
xi = (perstand / np.log(2)) * np.log(1.0 + 2.0 * wsstand / (tau_w_x + wsstand))
# xi is now shape (Nx,) and varies with x

lam = 1  # I don't know if this is correct.

plt.figure()
plt.plot(x, tau_w_x)
plt.title("Wall shear stress |tau_w(x)| (single Gaussian bump)")
plt.xlabel("x [cm]")
plt.ylabel("|tau_w| [g/(cm*s^2)]")
plt.grid(True)
plt.show()


# In[102]:


### Initial conditions ###
c_LDL_l[:, :] = Cl_in             # initial LDL in lumen 

# no cells or LDL in intima at t=0
c_LDL_i[:, :] = 0.0             
c_oxLDL_i[:, :] = 0.0
M[:, :] = 0.0
F[:, :] = 0.0


# In[104]:


### Explicit stability conditions time step ###

dt_adv = hx / Ul_max
dt_diff_l = 1.0 / (2 * Dl * (1/hx**2 + 1/hy1**2))
dt_diff_i = 1.0 / (2 * Di * (1/hx**2 + 1/hy2**2))
dt_react  = 1.0 / max(r_ox, kF * Cl_in)

dt = 0.5 * min(dt_adv, dt_diff_l, dt_diff_i, dt_react)

T  = 0.1
Nt = int(T / dt)

print(f"dt = {dt:.2e}, Nt = {Nt}")





# #### Velocity field in the lumen (Domain 1)
# 
# The velocity field is instead prescribed according to the project statement and the modeling assumptions of Calvez et al.
# 
# The velocity field is defined as a vector field  
# $\mathbf{v}(x,y) = (v_1(x,y), v_2(x,y))$,  
# where:
# 
# - $v_1(x,y)$ is the horizontal (x-direction) velocity component
# - $v_2(x,y)$ is the vertical (y-direction) velocity component
# 
# The velocity in the lumen is chosen as  
# $\mathbf{v}(x,y) = (v_{\max} P(y), 0)$,  
# meaning that:
# - the flow is unidirectional (along the artery axis)
# - there is no vertical velocity component
# - the velocity depends only on the y-coordinate
# 
# The function $P(y)$ is required to:
# - be quadratic
# - be positive inside the lumen
# - be 0 at the bottom and top walls of the lumen
# 
# Let the lumen occupy the vertical interval  
# $y \in [y_b, y_t]$,  
# where $y_b$ is the lower wall and $y_t$ is the upper wall.
# 
# The unique quadratic function satisfying these conditions is  
# $$
# P(y) = \frac{4 (y - y_b)(y_t - y)}{(y_t - y_b)^2}.
# $$
# 
# 
# This velocity profile corresponds to Poiseuille flow, which is the exact steady-state solution of the incompressible Navier–Stokes equations in a straight channel under a constant pressure gradient with no-slip boundary conditions.
# 
# Prescribing this velocity is therefore not a simplification, but an analytical reduction justified by:
# 
# - the separation of time scales between blood flow and inflammation,
# - the laminar nature of arterial flow,
# - the fixed geometry of the early-stage model.

# In[107]:


### Define velocity field in the lumen ###
def velocity_lumen(y):
    # Ly2 is top intima = bottom lumen
    # Ly1 + Ly2 is top lumen
    # v1(y) = C * (y - Ly2)(Ly2 + Ly1 - y) 
    # -> Quadratic is 0 at v1(Ly2) and v1(Ly2 + Ly1)
    # -> Gives (y - Ly2) and (y - (Ly2 + Ly1))
    return (4.0 * Ul_max * (y - Ly2) * (Ly2 + Ly1 - y)) / (Ly1**2)


# In[109]:


### Time loop ###
for n in range(Nt):  # n indexes the t


    # Copy so derivatives are computed from the old time level
    cl_n = c_LDL_l.copy()  # LDL lumen
    ci_n = c_LDL_i.copy()  # LDL intima
    cox_n = c_oxLDL_i.copy()  # ox-LDL intima
    M_n = M.copy()  # macrophages intima
    F_n = F.copy()  # foam cells intima
    S_n = S.copy()

    # Lumen: LDL transport (FD-BD-CD)
    # PDE:
    # Diffusion: CD, advection: BD, time: FD
    for j in range(1, Ny1 - 1):  # j indexes the y-direction

        vx = velocity_lumen(y1[j])  # velocity depends only on y

        for i in range(1, Nx - 1):  # i indexes the x-direction
            # Note that x and y are the otherway around here!!!! So that's why j is first and i second!!

            # Diffusion term
            # 2D Laplacian using CD (central differences) in x and y
            laplacian = (cl_n[j, i + 1] - 2 * cl_n[j, i] + cl_n[j, i - 1]) / hx * 2 + (
                        cl_n[j + 1, i] - 2 * cl_n[j, i] + cl_n[j - 1, i]) / hy1 * 2

            # Advection term using BD (backward difference), bc vx > 0
            advection = -vx * (cl_n[j, i] - cl_n[j, i - 1]) / hx

            # Time update using FD (forward difference)
            c_LDL_l[j, i] = cl_n[j, i] + dt * (Dl * laplacian + advection)

    ### External boundaries (no flux) lumen ###
    # Bottom boundary lumen is the interface with the top boundary intima, so not external!

    # Left boundary: inlet (x=0), using FD
    c_LDL_l[:, 0] = c_LDL_l[:, 1]  # c_LDL_l[:,0] = c_LDL_l[:,1]

    # Right boundary: outlet (x=Lx), using BD
    c_LDL_l[:, -1] = c_LDL_l[:, -2]  # c_LDL_l[:,Nx] = c_LDL_l[:,Nx-1]

    # Top boundary: arterial wall (y=Ly2+Ly1), using BD
    c_LDL_l[-1, :] = c_LDL_l[-2, :]  # c_LDL_l[Ny,:] = c_LDL_l[Ny-1,:]

    # Intima LDL
    for j in range(1, Ny2 - 1):
        for i in range(1, Nx - 1):
            # 2D CD approximation of diffusion LDL
            laplacian = (ci_n[j, i + 1] - 2 * ci_n[j, i] + ci_n[j, i - 1]) / hx * 2 + (
                        ci_n[j + 1, i] - 2 * ci_n[j, i] + ci_n[j - 1, i]) / hy2 * 2

            # FD for time update LDL
            c_LDL_i[j, i] = ci_n[j, i] + dt * (Di * laplacian - r_ox * ci_n[j, i])

    ### External boundaries (no flux) intima ###
    ## LDL intima
    # Left boundary (x=0), using FD
    c_LDL_i[:, 0] = c_LDL_i[:, 1]  # c_LDL_i[:,0] = c_LDL_i[:,1]

    # Right boundary (x=Lx), using BD
    c_LDL_i[:, -1] = c_LDL_i[:, -2]  # c_LDL_i[:,Nx] = c_LDL_i[:,Nx-1]

    # Bottom boundary: media (y=0), using FD
    c_LDL_i[0, :] = c_LDL_i[1, :]  # c_LDL_i[0,:] = c_LDL_li[1,:]

    for i in range(1, Nx - 1):
        flux = xi[i] * (cl_n[0, i] - ci_n[-1, i])

        # lumen loses LDL
        c_LDL_l[0, i] -= dt * flux / hy1

        # intima gains LDL
        c_LDL_i[-1, i] += dt * flux / hy2

    # Oxidized LDL
    for j in range(1, Ny2 - 1):
        for i in range(1, Nx - 1):
            # 2D CD approximation of diffusion oxLDL (same diffusion operator as LDL!!)
            lap = (cox_n[j, i + 1] - 2 * cox_n[j, i] + cox_n[j, i - 1]) / hx * 2 + (
                        cox_n[j + 1, i] - 2 * cox_n[j, i] + cox_n[j - 1, i]) / hy2 * 2

            # FD for time update oxLDL
            c_oxLDL_i[j, i] = cox_n[j, i] + dt * (d_ox * lap - kF * cox_n[j, i] * M_n[j, i] + r_ox * ci_n[j, i])
            # - consumption by macrophages + production from LDL

    # Macrophages
    for j in range(1, Ny2 - 1):
        for i in range(1, Nx - 1):
            # 2D CD approximation of diffusion macrophages (macrophages go through tissue and diffusion is simplest correct model)
            lap = (M_n[j, i + 1] - 2 * M_n[j, i] + M_n[j, i - 1]) / hx * 2 + (
                        M_n[j + 1, i] - 2 * M_n[j, i] + M_n[j - 1, i]) / hy2 * 2

            # Time step update macrophages: current conc. + diffusion - transformed into foam cells
            M[j, i] = M_n[j, i] + dt * (
                        dM * lap - kF * cox_n[j, i] * M_n[j, i])  # reaction removes macrophages if there is ox_LDL

    # Cytokines
    for j in range(1, Ny2 - 1):
        for i in range(1, Nx - 1):
            # 2D Laplacian CD (also diffusion used to describe movement cytokines, bc no better option)
            lap = (S_n[j, i + 1] - 2 * S_n[j, i] + S_n[j, i - 1]) / hx * 2 + (
                        S_n[j + 1, i] - 2 * S_n[j, i] + S_n[j - 1, i]) / hy2 * 2

            # Threshold, so cytokines are only produced if cox > cox_th, bc low ox-LDL does not trigger inflammation
            source = gamma * max(cox_n[j, i] - cox_th, 0.0) + kF * cox_n[j, i] * M_n[j, i]

            # Time step update cytokines: current conc. + diffusion - death rate + cytokine production (by inflammation and/or macrophage uptake)
            S[j, i] = S_n[j, i] + dt * (dS * lap - lam * S_n[j, i] + source)

    # Foam cells
    F += dt * (kF * cox_n * M_n)

    # Visualization
    if n % max(Nt // 10, 1) == 0:
        # Mesh for intima only
        X_i, Y_i = np.meshgrid(x, y2)

        plt.clf()
        plt.contourf(X_i, Y_i, np.log10(c_oxLDL_i + 1e-25), 30)
        plt.colorbar(label="log10(oxLDL)")

        # Interface line (lumen–intima)
        plt.plot([0, Lx], [Ly2, Ly2], 'k--', linewidth=1)

        plt.title(f"oxLDL in intima at t = {n * dt:.3e} s")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.01)

plt.show()

            
    

    


# In[ ]:




