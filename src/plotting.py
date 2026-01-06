
import numpy as np
import matplotlib.pyplot as plt
import params as p


def maybe_plot(n, dt, x, y1, y2, c_LDL_i, c_LDL_l):
    # Visualization
    if n % 50 == 0:
        y_all = np.concatenate((y2, y1[1:]))
        u_all = np.vstack((c_LDL_i, c_LDL_l[1:,:]))

        X, Y = np.meshgrid(x, y_all)

        plt.clf()
        plt.contourf(X, Y, u_all, 30)
        plt.colorbar(label="LDL concentration")
        plt.plot([0, p.Lx], [p.Ly2, p.Ly2], 'k--', linewidth=1)
        plt.title(f"t = {n*dt:.3e} s")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.01)
