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
from ase.visualize import view

db = connect('structures/reference_data.db')
for i,row in enumerate(db.select()):
    atoms = row.toatoms()
    E_mix = row.mixing_energy
    print(i,row.symbols, E_mix)
    if(i==3):
        view(atoms)

# %%

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



# %% [markdown]
## Task 2: Cutoff selection for a pair cluster-expansion  (5p)


# %%
from icet import ClusterSpace

from sklearn import linear_model
from sklearn.model_selection import KFold

x=[]

# setup CS
cutoffs = [8]
prim = db.get(1).toatoms()
cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=['Ag', 'Pd'])

for i,row in enumerate(db.select()):
    # get cluster-vector for a given atoms object
    atoms = row.toatoms()
    x.append(cs.get_cluster_vector(atoms))

x=np.array(x)
print(x.shape)

# %%
def run_OLS_fit(A, y):
    ols = linear_model.LinearRegression(fit_intercept=False)
    ols.fit(A, y)
    return ols.coef_


def compute_mse(A, y, parameters):
    y_predicted = np.dot(A, parameters)
    dy = y - y_predicted
    mse = np.mean(dy**2)
    return mse


def compute_rmse(A, y, parameters):
    return np.sqrt(compute_mse(A, y, parameters))


def get_aic_bic(A, y, parameters):

    n_samples = len(y)
    n_parameters = len(parameters)
    mse = compute_mse(A, y, parameters)

    aic = n_samples * np.log(mse) + 2 * n_parameters
    bic = n_samples * np.log(mse) + n_parameters * np.log(n_samples)

    return -aic, -bic

def run_cv(A, y):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_train = []
    rmse_test = []
    for train_inds, test_inds in cv.split(A):
        A_train = A[train_inds]
        y_train = y[train_inds]
        A_test = A[test_inds]
        y_test = y[test_inds]

        parameters = run_OLS_fit(A_train, y_train)
        rmse_train.append(compute_rmse(A_train, y_train, parameters))
        rmse_test.append(compute_rmse(A_test, y_test, parameters))

    data = dict()
    data['rmse_train'] = np.mean(rmse_train)
    data['rmse_train_std'] = np.std(rmse_train)
    data['rmse_validation'] = np.mean(rmse_test)
    data['rmse_validation_std'] = np.std(rmse_test)
    return data

def full_analysis(A, y):

    # run cv
    cv_data = run_cv(A, y)

    # final fit
    parameters = run_OLS_fit(A, y)

    # compute AIC/BIC
    aic, bic = get_aic_bic(A, y, parameters)

    # finalize data
    data = dict(aic=aic, bic=bic)
    data.update(cv_data)
    return data

#%%
# generate data
np.random.seed(42)
cutoffs = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]
cutoffs = [[x] for x in range(1,20)]
data_list = []

aic_list=[]
bic_list=[]
rmse_train_list=[]
rmse_valid_list=[]
std_train_list=[]
std_valid_list=[]
mse_list = []

for cutoff in cutoffs:
    print('Cutoff= ', cutoff[0])
    x=[]

    # setup CS
    #cutoffs = [8]
    prim = db.get(1).toatoms()
    cs = ClusterSpace(prim, cutoffs=cutoff, chemical_symbols=['Ag', 'Pd'])

    for i,row in enumerate(db.select()):
        # get cluster-vector for a given atoms object
        atoms = row.toatoms()
        x.append(cs.get_cluster_vector(atoms))

    x=np.array(x)

    #N, M = 200, 100
    A = x
    #parameters_true = np.random.normal(0, 1, (M, ))
    #noise = np.random.normal(0, 0.05, (N, ))
    y = np.array(E_mix_list)

    # OLS fit
    data = full_analysis(A, y)
    data_list.append(data)
    # for key, val in data.items():
    #     print(f'{key:20} : {val:11.5f}')
    # print('-----------------------\n')

    aic_list.append(data.get('aic'))
    bic_list.append(data.get('bic'))
    rmse_train_list.append(data.get('rmse_train'))
    rmse_valid_list.append(data.get('rmse_validation'))
    std_train_list.append(data.get('rmse_train_std'))
    std_valid_list.append(data.get('rmse_validation_std'))

    parameters = run_OLS_fit(A, y)
    mse = compute_mse(A, y, parameters)
    mse_list.append(mse)
# %%

fig_IC=plt.figure()
plt.plot(cutoffs,aic_list,label='AIC')
plt.plot(cutoffs,bic_list,label='BIC')

plt.xlabel('cutoff [Å]')
plt.ylabel('IC')
plt.legend()

fig_rmse=plt.figure()
plt.plot(cutoffs,rmse_train_list,label='Training')
plt.plot(cutoffs,rmse_valid_list,label='Validation')

plt.xlabel('cutoff [Å]')
plt.ylabel('RMSE')
plt.legend()

fig_std=plt.figure()
plt.plot(cutoffs,std_train_list,label='Training')
plt.plot(cutoffs,std_valid_list,label='Validation')

plt.xlabel('cutoff [Å]')
plt.ylabel('STD')
plt.legend()

fig_mse = plt.figure('MSE')
plt.plot(cutoffs,mse_list)
# %% [markdown]
## Task 3:  Feature selection  (5p)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
print(np.std(x, axis=0))
x_stand=scaler.transform(x)
print(np.std(x_stand, axis=0))
print(np.mean(x_stand, axis=0))
#%%
def run_cv_Lasso(A, y,alpha):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_train = []
    rmse_test = []

    lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    
    for train_inds, test_inds in cv.split(A):
        A_train = A[train_inds]
        y_train = y[train_inds]
        A_test = A[test_inds]
        y_test = y[test_inds]

        lasso.fit(A_train, y_train)
        parameters = lasso.coef_

        rmse_train.append(compute_rmse(A_train, y_train, parameters))
        rmse_test.append(compute_rmse(A_test, y_test, parameters))

    data = dict()
    data['rmse_train'] = np.mean(rmse_train)
    data['rmse_train_std'] = np.std(rmse_train)
    data['rmse_validation'] = np.mean(rmse_test)
    data['rmse_validation_std'] = np.std(rmse_test)
    return data

def get_aic_bic_sparse(A, y, parameters):

    n_samples = len(y)
    n_parameters = sum(parameters!=0)
    mse = compute_mse(A, y, parameters)

    aic = n_samples * np.log(mse) + 2 * n_parameters
    bic = n_samples * np.log(mse) + n_parameters * np.log(n_samples)

    return -aic, -bic
#%%
alpha_list = np.linspace(0.2,10,100)

aic_list=[]
bic_list=[]
rmse_train_list=[]
rmse_valid_list=[]
std_train_list=[]
std_valid_list=[]
mse_list = []
nbr_params_list=[]

y_stand = y
# y_stand=(y-np.mean(y))/np.std(y)
#A_stand=(x-np.mean(y))/np.std(y)

A=x_stand

for alpha in alpha_list:
    
    # print(sum(params!=0))

    cv_data = run_cv_Lasso(A, y_stand, alpha)


    lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(A, y_stand)
    params = lasso.coef_
    aic, bic = get_aic_bic_sparse(A, y_stand, params)

    nbr_params_list.append(sum(params!=0))
    aic_list.append(aic)
    bic_list.append(bic)
    rmse_train_list.append(cv_data.get('rmse_train'))
    rmse_valid_list.append(cv_data.get('rmse_validation'))
    std_train_list.append(cv_data.get('rmse_train_std'))
    std_valid_list.append(cv_data.get('rmse_validation_std'))

#%%
fig_rmse=plt.figure()
plt.plot(alpha_list,rmse_train_list,label='Training')
plt.plot(alpha_list,rmse_valid_list,label='Validation')

plt.xlabel(r'$\alpha$ [a.u.]')
plt.ylabel('RMSE')
plt.legend()
# plt.xlim(0.5,10)
# plt.ylim(35,45)

fig_rmse_vs_params=plt.figure()
plt.plot(nbr_params_list,rmse_train_list,label='Training')
plt.plot(nbr_params_list,rmse_valid_list,label='Validation')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('RMSE')
plt.legend()

fig_IC=plt.figure()
plt.plot(nbr_params_list,aic_list,label='AIC')
plt.plot(nbr_params_list,bic_list,label='BIC')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('IC')
plt.legend()

# %%

lmb = 100
ardr = linear_model.ARDRegression(threshold_lambda=lmb, fit_intercept=False)
ardr.fit(x_stand, y)
params = ardr.coef_
print(sum(params!=0))

# %%
def run_cv_ARDR(A, y,threshold_lambda):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_train = []
    rmse_test = []

    ardr = linear_model.ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=False)
    
    for train_inds, test_inds in cv.split(A):
        A_train = A[train_inds]
        y_train = y[train_inds]
        A_test = A[test_inds]
        y_test = y[test_inds]

        ardr.fit(A_train, y_train)
        parameters = ardr.coef_

        rmse_train.append(compute_rmse(A_train, y_train, parameters))
        rmse_test.append(compute_rmse(A_test, y_test, parameters))

    data = dict()
    data['rmse_train'] = np.mean(rmse_train)
    data['rmse_train_std'] = np.std(rmse_train)
    data['rmse_validation'] = np.mean(rmse_test)
    data['rmse_validation_std'] = np.std(rmse_test)
    return data

# %%
# lambda_list = np.append(np.linspace( 0, 0.9,10),np.linspace(1,10000,50))
lambda_list=np.append(np.linspace( 0, 0.5,100),np.linspace(1,200,20))
aic_list=[]
bic_list=[]
rmse_train_list=[]
rmse_valid_list=[]
std_train_list=[]
std_valid_list=[]
mse_list = []
nbr_params_list=[]

y_stand = y
# y_stand=(y-np.mean(y))/np.std(y)
#A_stand=(x-np.mean(y))/np.std(y)

A=x_stand
for threshold_lambda in lambda_list:
    
    # print(sum(params!=0))

    cv_data = run_cv_ARDR(A, y_stand, threshold_lambda)


    ardr = linear_model.ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=False)
    ardr.fit(A, y_stand)
    params = ardr.coef_
    aic, bic = get_aic_bic_sparse(A, y_stand, params)
    new = 
    nbr_params_list.append(sum(params!=0))
    aic_list.append(aic)
    bic_list.append(bic)
    rmse_train_list.append(cv_data.get('rmse_train'))
    rmse_valid_list.append(cv_data.get('rmse_validation'))
    std_train_list.append(cv_data.get('rmse_train_std'))
    std_valid_list.append(cv_data.get('rmse_validation_std'))


# %%
fig_rmse=plt.figure()
plt.plot(lambda_list,rmse_train_list,label='Training')
plt.plot(lambda_list,rmse_valid_list,label='Validation')

plt.xlabel(r'$\lambda$ [a.u.]')
plt.ylabel('RMSE')
plt.legend()
plt.xlim(0,1)
# plt.ylim(35,45)

fig_lambda_params=plt.figure()
plt.semilogx(lambda_list,nbr_params_list)

plt.xlabel(r'$\lambda$ [a.u.]')
plt.ylabel('Number of parameters [a.u.]')
# plt.xlim(0,1)

fig_rmse_vs_params=plt.figure()
plt.plot(nbr_params_list,rmse_train_list,label='Training')
plt.plot(nbr_params_list,rmse_valid_list,label='Validation')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('RMSE')
plt.legend()

fig_IC=plt.figure()
plt.plot(nbr_params_list,aic_list,label='AIC')
plt.plot(nbr_params_list,bic_list,label='BIC')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('IC')
plt.legend()

# %% [markdown]

## Task 4: Bayesian Cluster expansion   (7p)
#  
# %%

# Get new cluster space with cutoffs

# define priors and likelihood

# Sample posterior (for the given cluster candidates), visualize

# Do OLS with new cutoff, find lowest energy configuration among candidates

# Calculate lowest energy candidate for each sample,
# save lowest energy for each sample.

#Frequency of being the lowest is converted to probability of the same.

# Plot the distribution of lowest energies.

