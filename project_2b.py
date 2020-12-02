#%%
import ase
# import asap3
import GPy

from ase.constraints import FixAtoms, FixedLine
from ase.optimize import BFGS
from ase.visualize import view
from asap3 import EMT
# from ase.calculators.emt import EMT
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

def calc_surface_energy(surface):
    # add adatom
    surface_with_adatom = surface.copy()

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
E_surface = calc_surface_energy(surface)
size = 100
xmax=16.65653
ymax=2.884996
E=np.empty((size,size))

x = np.arange(xmax/(size+1),xmax,xmax/(size+1))
y = np.arange(ymax/(size+1),ymax,ymax/(size+1))
z = surface.get_positions()[:, 2].max() + 3

#%%

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
plt.savefig(fname='heatmap.pdf')

#%%
def min_fun(pos):
    x=pos[0]%xmax
    y=pos[1]%ymax
    return calculate_adatom_energy(surface,(x, y, z))

#%%
num_runs=200
start=np.random.rand(2,num_runs)
startx=start[0]*xmax
starty=start[1]*ymax
E_local = np.empty(num_runs)
x_local = np.empty((num_runs,2))
result_local = []
for i in range(num_runs):
    # minimize(min_fun, [startx[i], starty[i]])
    result_local.append(minimize(lambda Pos: calculate_adatom_energy(surface, (Pos[0],Pos[1],z)), [startx[i], starty[i]]))
    E_local[i]=result_local[i].fun
    x_local[i]=result_local[i].x
    print("i=",i)
# %%
plt.hist(E_local)
plt.show()
plt.scatter(x_local[:,0]%xmax,x_local[:,1]%ymax)
plt.show()
a,b,c,d=plt.hist2d(x_local[:,0]%xmax,x_local[:,1]%ymax,bins=50)
print(a[a!=0])
plt.colorbar()
plt.show()
# %%
plt.contourf(x,y,E, cmap='hot')
#plt.scatter((x_local[:,1]%ymax)*xmax/ymax,(x_local[:,0]%xmax)*ymax/xmax)
plt.scatter(x_local[:,0],x_local[:,1])
plt.xlim((0,xmax))
plt.ylim((0,ymax))
#%%
plt.contourf(x,y,np.transpose(E)-E_surface, cmap='hot')
plt.colorbar()
plt.scatter(x_local[:,0]%xmax,x_local[:,1]%ymax)


#%% Task 3, Start samples
np.random.seed(123)
num_runs=5
start=np.random.rand(num_runs,2)
startx=start[0]*xmax
starty=start[1]*ymax
E_data = np.empty((num_runs,1))
x_data = start
result_task3 = []
for i in range(num_runs):
    # minimize(min_fun, [startx[i], starty[i]])
    #result_task3.append(minimize(lambda Pos: calculate_adatom_energy(surface, (Pos[0],Pos[1],z)), [startx[i], starty[i]]))
    E_data[i]=calculate_adatom_energy(surface, (start[i,0],start[i,1],z))
    #x_data[i,:]=start[i]


# %%
beta = 3
def neg_A(x, model):
    mu,sigma = model.predict(np.array(x,ndmin=2))
    return (mu-beta*sigma)[0]

def new_sample(model):
    num_runs=20
    start=np.random.rand(num_runs,2)*np.array([xmax,ymax])
    #start=np.random.rand(2,num_runs)*np.array([xmax,ymax])
    startx=start[0]*xmax
    starty=start[1]*ymax
    E_data = np.empty(num_runs)
    x_data = np.empty((num_runs,2))
    result_task3 = []
    for i in range(num_runs):
        result_task3.append(minimize(neg_A, start[i],args=model,bounds=[(0,xmax),(0,ymax)]))
        E_data[i]=result_task3[i].fun
        x_data[i]=result_task3[i].x
    #model.optimize_restarts()

    return start[np.argmin(E_data)], calculate_adatom_energy(surface, (start[np.argmin(E_data),0],start[np.argmin(E_data),1],z))
# %%
np.random.seed(2)
k1 = GPy.kern.RBF(input_dim=2)
k1['lengthscale'].constrain_bounded(0.1, 5)
# k1['lengthscale'].set_prior(GPy.priors.Gamma(a=2, b=1))
k2 = GPy.kern.Bias(input_dim=2)
kernel = k1 + k2

num_runs3=10

for i in range(num_runs3):
    model = GPy.models.GPRegression(x_data, E_data, kernel)
    model.optimize()
    
    x_new, E_new=new_sample(model)
    x_data=np.append(x_data,np.array(x_new,ndmin=2),axis=0)
    E_data=np.append(E_data,np.array(E_new,ndmin=2),axis=0)


#print(model)
#model.
#mu,sigma = model.predict(np.array((1,1),ndmin=2))

plt.scatter(x_data[:,0],x_data[:,1])

# %%
1+1
# %%
