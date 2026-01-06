# NEW: split from original script into a module

import numpy as np
import params as p


# In[96]:


def build_grid():
    # Spatial discretization
    hx = p.Lx / (p.Nx - 1)
    hy1 = p.Ly1 / (p.Ny1 -1)
    hy2 = p.Ly2 / (p.Ny2 -1)

    # Arrays
    x = np.linspace(0.0, p.Lx, p.Nx)
    y2 = np.linspace(0.0, p.Ly2, p.Ny2)        # intima (lower domain (2))
    y1 = np.linspace(p.Ly2, p.Ly2 + p.Ly1, p.Ny1)  # lumen  (upper domain (1))

    return x, y1, y2, hx, hy1, hy2
