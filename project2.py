#%%

import ase
import icet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import emcee
import corner
from scipy.stats import gamma, invgamma, t, norm, norminvgauss, mode
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style("darkgrid")
sns.set_palette("deep")
sns.set(font='sans-serif')
%matplotlib inline
plt.rcParams['figure.dpi'] = 140

np.random.seed(123)

# %% [markdown]
# # Project 2a: Alloy cluster expansions
# ### *Sebastian Holmin, Erik Andersson, 2020*
# # Task 1: Collect and plot the data (0.5p)

# %%
from ase.db import connect

db = connect('structures/reference_data.db')
for i,row in enumerate(db.select()):
    atoms = row.toatoms()
    E_mix = row.mixing_energy
    print(i,row.symbols, E_mix)
    if(i==3):
        view(atoms)

# %%
from ase.visualize import view
view(db.select().toatoms())

# %%
E_mix_list= []
Pd_desity_list = []
for i,row in enumerate(db.select()):
    atoms = row.toatoms()
    atomic_numbers = atoms.get_atomic_numbers()
    E_mix = row.mixing_energy
    Pd_desity_list.append(sum(atomic_numbers==46)/len(atomic_numbers))
    E_mix_list.append(E_mix)
plt.scatter(Pd_desity_list,E_mix_list)
# %%
atomic_numbers = atoms.get_atomic_numbers()
print(sum(atomic_numbers==46)/len(atomic_numbers))
# %%

from icet import ClusterSpace

from sklearn import linear_model

# setup CS
cutoffs = [8]
prim = db.get(1).toatoms()
cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=['Ag', 'Pd'])

# get cluster-vector for a given atoms object
atoms = db.get(3).toatoms()
x = cs.get_cluster_vector(atoms)
print(x.shape)




# %%
