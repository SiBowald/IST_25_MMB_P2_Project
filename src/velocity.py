
import params as p


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
    return (4.0 * p.Ul_max * (y - p.Ly2) * (p.Ly2 + p.Ly1 - y)) / (p.Ly1**2)

# NEW: derivative of the prescribed lumen velocity profile
def d_velocity_lumen_dy(y):
    # derivative of:
    # v(y) = (4*Ul_max*(y-Ly2)*(Ly2+Ly1-y))/Ly1^2
    return (4.0 * p.Ul_max * (2.0 * p.Ly2 + p.Ly1 - 2.0 * y)) / (p.Ly1**2)


# NEW: wall shear stress magnitude at the endothelium (bottom wall of lumen, y = Ly2)
def wall_shear_stress_endothelium():
    yb = p.Ly2
    return p.nu * abs(d_velocity_lumen_dy(yb))
