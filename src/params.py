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
