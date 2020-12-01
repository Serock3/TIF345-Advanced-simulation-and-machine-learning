#%%
import ase
# import asap3
import GPy

from ase.constraints import FixAtoms, FixedLine
from ase.optimize import BFGS
from ase.visualize import view
# from asap3 import EMT
from ase.calculators.emt import EMT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.gridspec import GridSpec
# import emcee
# import corner
from scipy.stats import gamma, invgamma, t, norm, norminvgauss, mode
from scipy.optimize import minimize
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style("darkgrid")
sns.set_palette("deep")
sns.set(font='sans-serif')
%matplotlib inline
plt.rcParams['figure.dpi'] = 140

np.random.seed(123)
#%%
def calculate_adatom_energy(surface, position):
    """Adds adatom to the input surface configuration at the given position
    and relaxes the ad-atoms z-cooridnate.

    Parameters
    ----------
    bare_surface
        surface configuration without adatom
    position
        position (x, y, z) at which to insert adatom in Ångström

    Returns
    -------
    tuple comprising the energy of the adatom configuration
    """
    # add adatom
    surface_with_adatom = surface.copy()
    surface_with_adatom.append(ase.Atom('Au', position))

    # attach calculator
    calc = EMT()
    surface_with_adatom.set_calculator(calc)

    # apply constraints
    constraints = []
    c = FixAtoms(indices=list(range(len(surface_with_adatom) - 1)))
    constraints.append(c)
    c = FixedLine(-1, [0, 0, 1])
    constraints.append(c)

    # relax configuration
    surface_with_adatom.set_constraint(constraints)
    dyn = BFGS(surface_with_adatom, logfile=None)
    dyn.run(fmax=0.02, steps=200)

    energy = surface_with_adatom.get_potential_energy()
    return energy

#%% Task 1: Analyzing the PES (1.5p)
from ase.io import read
surface = read('structures/surface_supercell.xyz')
size = 50
xmax=16.65653
ymax=2.884996
E=np.empty((size,size))

x = np.arange(xmax/(size+1),xmax,xmax/(size+1))
y = np.arange(ymax/(size+1),ymax,ymax/(size+1))
z = surface.get_positions()[:, 2].max() + 3

for xi in range(x.shape[0]):
    for yi in range(y.shape[0]):
        E[xi,yi] = calculate_adatom_energy(surface,(x[xi], y[yi],z))
    print("xi = ",xi," done")

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x,y = np.meshgrid(x, y)
# surf = ax.heatmap(x, y, E, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()
plt.contourf(x,y,E, cmap='hot')
plt.colorbar()
plt.savefig()
#%%
k1 = GPy.kern.RBF(input_dim=2)
k1['lengthscale'].constrain_bounded(0.1, 5)
k2 = GPy.kern.Bias(input_dim=2)
kernel = k1 + k2
model = GPy.models.GPRegression(x_data, y_data, kernel)
model.optimize()
print(model)
#model.optimize_restarts()
#%%
def min_fun(pos):
    return calculate_adatom_energy(surface,(pos[0], pos[1], z))

#%%
num_runs=3
start=np.random.rand(2,num_runs)
startx=start[0]*xmax
starty=start[1]*ymax
E_local = np.empty(num_runs)
for i in range(num_runs):
    # minimize(min_fun, [startx[i], starty[i]])
    minimize(lambda Pos: calculate_adatom_energy(surface, (Pos[0],Pos[1],z)), [startx[i], starty[i]])


# %%
