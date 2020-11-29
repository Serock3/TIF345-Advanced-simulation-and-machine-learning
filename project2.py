# %%

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
from icet import ClusterSpace
from ase.visualize import view
from ase.db import connect
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

db = connect('structures/reference_data.db')
for i, row in enumerate(db.select()):
    atoms = row.toatoms()
    E_mix = row.mixing_energy
    print(i, row.symbols, E_mix)
    if(i == 3):
        view(atoms)

# %%
E_mix_list = []
Pd_desity_list = []
for i, row in enumerate(db.select()):
    atoms = row.toatoms()
    atomic_numbers = atoms.get_atomic_numbers()
    E_mix = row.mixing_energy
    Pd_desity_list.append(sum(atomic_numbers == 46)/len(atomic_numbers))
    E_mix_list.append(E_mix)
plt.scatter(Pd_desity_list, E_mix_list, 4, 'black')
plt.xlabel('Pd concentration [a.u.]')
plt.ylabel('Energy per atom [meV]')

plt.savefig('energy_vs_Pd_cons.pdf')

# %%
atomic_numbers = atoms.get_atomic_numbers()
print(sum(atomic_numbers == 46)/len(atomic_numbers))


# %% [markdown]
# Task 2: Cutoff selection for a pair cluster-expansion  (5p)


# %%


x = []

# setup CS
cutoffs = [8]
prim = db.get(1).toatoms()
cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=['Ag', 'Pd'])

for i, row in enumerate(db.select()):
    # get cluster-vector for a given atoms object
    atoms = row.toatoms()
    x.append(cs.get_cluster_vector(atoms))

x = np.array(x)
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
    print(np.std(parameters))
    # compute AIC/BIC
    aic, bic = get_aic_bic(A, y, parameters)

    # finalize data
    data = dict(aic=aic, bic=bic)
    data.update(cv_data)
    return data


# %%
# generate data
np.random.seed(42)
cutoffs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
cutoffs = [[x] for x in range(1, 20)]
data_list = []

aic_list = []
bic_list = []
rmse_train_list = []
rmse_valid_list = []
std_train_list = []
std_valid_list = []
mse_list = []

for cutoff in cutoffs:
    print('Cutoff= ', cutoff[0])
    x = []

    # setup CS
    # cutoffs = [8]
    prim = db.get(1).toatoms()
    cs = ClusterSpace(prim, cutoffs=cutoff, chemical_symbols=['Ag', 'Pd'])

    for i, row in enumerate(db.select()):
        # get cluster-vector for a given atoms object
        atoms = row.toatoms()
        x.append(cs.get_cluster_vector(atoms))

    x = np.array(x)
    if cutoff[0] == 6:
        print('NUM PARAMETERS AT 6Å', x.shape[1])
    # N, M = 200, 100
    A = x
    # parameters_true = np.random.normal(0, 1, (M, ))
    # noise = np.random.normal(0, 0.05, (N, ))
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

fig_IC = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.plot(cutoffs, aic_list, label='AIC')
plt.plot(cutoffs, bic_list, label='BIC')


plt.xlabel('cutoff [Å]')
plt.ylabel('IC')
plt.legend()
plt.tight_layout()
plt.savefig('cutoff_IC.pdf')

fig_rmse = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.plot(cutoffs, rmse_train_list, label='Training')
plt.plot(cutoffs, rmse_valid_list, label='Validation')

plt.xlabel('cutoff [Å]')
plt.ylabel('RMSE [meV]')
plt.legend()
plt.tight_layout()
plt.savefig('cutoff_RMSE.pdf')

fig_std = plt.figure()
plt.plot(cutoffs, std_train_list, label='Training')
plt.plot(cutoffs, std_valid_list, label='Validation')

plt.xlabel('cutoff [Å]')
plt.ylabel('STD [meV]')
plt.legend()

fig_mse = plt.figure('MSE')
plt.plot(cutoffs, mse_list)
# %% [markdown]
# Task 3:  Feature selection  (5p)

# %%
scaler = StandardScaler()
scaler.fit(x)
print(np.std(x, axis=0))
x_stand = scaler.transform(x)
print(np.std(x_stand, axis=0))
print(np.mean(x_stand, axis=0))
# %%


def run_cv_Lasso(A, y, alpha):
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
    n_parameters = sum(parameters != 0)
    mse = compute_mse(A, y, parameters)

    aic = n_samples * np.log(mse) + 2 * n_parameters
    bic = n_samples * np.log(mse) + n_parameters * np.log(n_samples)

    return -aic, -bic


# %%
alpha_list = np.linspace(0.2, 10, 100)

aic_list = []
bic_list = []
rmse_train_list = []
rmse_valid_list = []
std_train_list = []
std_valid_list = []
mse_list = []
nbr_params_list = []

y_stand = y
# y_stand=(y-np.mean(y))/np.std(y)
# A_stand=(x-np.mean(y))/np.std(y)

A = x_stand

save_params = []

for alpha in alpha_list:

    # print(sum(params!=0))

    cv_data = run_cv_Lasso(A, y_stand, alpha)

    lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(A, y_stand)
    params = lasso.coef_
    aic, bic = get_aic_bic_sparse(A, y_stand, params)

    nbr_params_list.append(sum(params != 0))

    if sum(params != 0) == 4:
        # save_params.append(np.where(params!=0))
        save_params.append(params[1:5])

    print(np.std(scaler.transform(params.reshape(1, 62))))
    aic_list.append(aic)
    bic_list.append(bic)
    rmse_train_list.append(cv_data.get('rmse_train'))
    rmse_valid_list.append(cv_data.get('rmse_validation'))
    std_train_list.append(cv_data.get('rmse_train_std'))
    std_valid_list.append(cv_data.get('rmse_validation_std'))

# %%
fig_rmse = plt.figure()
plt.plot(alpha_list, rmse_train_list, label='Training')
plt.plot(alpha_list, rmse_valid_list, label='Validation')

plt.xlabel(r'$\alpha$ [a.u.]')
plt.ylabel('RMSE')
plt.legend()
# plt.xlim(0.5,10)
# plt.ylim(35,45)

fig_rmse_vs_params = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.scatter(nbr_params_list, rmse_train_list, 4, label='Training')
plt.scatter(nbr_params_list, rmse_valid_list, 4, label='Validation')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('RMSE [meV]')
plt.legend()
plt.tight_layout()
plt.savefig('Lasso_RMSE.pdf')

fig_IC = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.scatter(nbr_params_list, aic_list, 4, label='AIC')
plt.scatter(nbr_params_list, bic_list, 4, label='BIC')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('IC')
plt.legend()
plt.tight_layout()
plt.savefig('Lasso_IC.pdf')

# %%

lmb = 100
ardr = linear_model.ARDRegression(threshold_lambda=lmb, fit_intercept=False)
ardr.fit(x_stand, y)
params = ardr.coef_
print(sum(params != 0))

# %%


def run_cv_ARDR(A, y, threshold_lambda):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_train = []
    rmse_test = []

    ardr = linear_model.ARDRegression(
        threshold_lambda=threshold_lambda, fit_intercept=False)

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
lambda_list = np.append(np.linspace(0, 0.5, 100), np.linspace(1, 200, 20))
aic_list = []
bic_list = []
rmse_train_list = []
rmse_valid_list = []
std_train_list = []
std_valid_list = []
mse_list = []
nbr_params_list = []

y_stand = y
# y_stand=(y-np.mean(y))/np.std(y)
# A_stand=(x-np.mean(y))/np.std(y)

save_params = []

A = x_stand
for threshold_lambda in lambda_list:

    # print(sum(params!=0))

    cv_data = run_cv_ARDR(A, y_stand, threshold_lambda)
    ardr = linear_model.ARDRegression(
        threshold_lambda=threshold_lambda, fit_intercept=False)
    ardr.fit(A, y_stand)
    params = ardr.coef_
    aic, bic = get_aic_bic_sparse(A, y_stand, params)

    nbr_params_list.append(sum(params != 0))

    if sum(params != 0) == 4:
        save_params.append(params[1:5])
        # save_params.append(np.where(params!=0))

    aic_list.append(aic)
    bic_list.append(bic)
    rmse_train_list.append(cv_data.get('rmse_train'))
    rmse_valid_list.append(cv_data.get('rmse_validation'))
    std_train_list.append(cv_data.get('rmse_train_std'))
    std_valid_list.append(cv_data.get('rmse_validation_std'))


# %%
fig_rmse = plt.figure()
plt.plot(lambda_list, rmse_train_list, label='Training')
plt.plot(lambda_list, rmse_valid_list, label='Validation')

plt.xlabel(r'$\lambda$ [a.u.]')
plt.ylabel('RMSE')
plt.legend()
plt.xlim(0, 1)
# plt.ylim(35,45)

fig_lambda_params = plt.figure()
plt.semilogx(lambda_list, nbr_params_list)

plt.xlabel(r'$\lambda$ [a.u.]')
plt.ylabel('Number of parameters [a.u.]')
# plt.xlim(0,1)

fig_rmse_vs_params = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.scatter(nbr_params_list, rmse_train_list, 4, label='Training')
plt.scatter(nbr_params_list, rmse_valid_list, 4, label='Validation')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('RMSE [meV]')
plt.legend()
plt.tight_layout()
plt.savefig('ARDR_RMSE.pdf')

fig_IC = plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.scatter(nbr_params_list, aic_list, 4, label='AIC')
plt.scatter(nbr_params_list, bic_list, 4, label='BIC')

plt.xlabel(r'Number of parameters [a.u.]')
plt.ylabel('IC')
plt.legend()
plt.tight_layout()
plt.savefig('ARDR_IC.pdf')

# %% [markdown]

# Task 4: Bayesian Cluster expansion   (7p)
#
# %%

# Get new cluster space with cutoffs
x = []

# setup CS
cutoffs = [12.0, 6.0]
prim = db.get(1).toatoms()
cs = ClusterSpace(prim, cutoffs=cutoffs, chemical_symbols=['Ag', 'Pd'])

for i, row in enumerate(db.select()):
    # get cluster-vector for a given atoms object
    atoms = row.toatoms()
    x.append(cs.get_cluster_vector(atoms))

x = np.array(x)

# %%
# define priors and likelihood


def mean_mode_2_IG_alpha_beta(mean, mode):
    alpha = (mode + mean)/(mean-mode)
    beta = (2 * mode * mean)/(mean-mode)
    return alpha, beta


a0_sig, b0_sig = mean_mode_2_IG_alpha_beta(100, 1)
a0_alpha, b0_alpha = mean_mode_2_IG_alpha_beta(2500, 25)


def log_prior(j, sigma2, alpha2, nP):
    return -0.5*nP*np.log(alpha2)-0.5*np.sum(j**2)/alpha2 + invgamma.logpdf(sigma2, a=a0_sig, scale=b0_sig) + invgamma.logpdf(alpha2, a=a0_alpha, scale=b0_alpha)


def log_likelihood(model, sigma2, data):
    return -np.sum((model-data)**2)/(2*sigma2)-0.5*len(data)*np.log(sigma2)


def log_posterior(params, A, data, nP):
    j = params[:nP]
    sigma2 = params[nP]
    alpha2 = params[nP+1]

    model = np.matmul(A, j)

    lp = log_prior(j, sigma2, alpha2, nP)
    if not np.isfinite(lp):
        return -np.inf

    return log_likelihood(model, sigma2, data)+lp


# %% Sample posterior (for the given cluster candidates), visualize
ndim, nwalkers = x.shape[1]+2, 100
# start_pos = [70,0,0.1] + [1e-2,1e-2,1e-5]*np.random.randn(nwalkers, ndim)
start_pos = [0]*x.shape[1]+[10]*np.random.randn(nwalkers, ndim-2)
start_pos = np.append(start_pos, [7, 30]+[1e-1]
                      * np.random.randn(nwalkers, 2), axis=1)

steps = 5000


sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, args=(x, y, ndim-2))
sampler.run_mcmc(start_pos, steps, progress=True)

# %%


def simple_mcmc_analysis(sampler, par, label, burn_in, chain_from_file=False):

    if not chain_from_file:
        print(
            f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):0.3f}')
    # discard the first 'burn_in' samples

    # thinning means that you only keep every nth sample. E.g. thinning=10 -> keep every 10th sample.
    # This can be useful for reducing long autocorrelation lenghts in a chain. However, thinning is expensive.
    # A thinned chain must be run E.g. 10x longer to reach the desired number of samples.
    # One can argue that thinning is not an advantageous strategy. So keep thinning = 1
    thinning = 1
    flat_mcmc_samples = sampler.get_chain(
        discard=burn_in, thin=thinning, flat=True)
    print(f'Discarding {nwalkers*burn_in} steps as burn-in')
    print(f'Chain length:{len(flat_mcmc_samples)}')

    fig1 = plt.figure()
    plt.plot(flat_mcmc_samples[:, par], color='gray', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel(label)
    plt.xlim(0, len(flat_mcmc_samples))

    return flat_mcmc_samples


# %%
burn_in = 3000

flat_mcmc_samples = simple_mcmc_analysis(
    sampler, par=0, label=f'$H_0$', burn_in=burn_in)

fig = corner.corner(flat_mcmc_samples[:, 33:35], show_titles=True)

# %%


# %% Do OLS with new cutoff, find lowest energy configuration among candidates
db_gs = connect('structures/ground_state_candidates.db')

x_gs = []

for i, row in enumerate(db_gs.select()):
    atoms = row.toatoms()
    x_gs.append(cs.get_cluster_vector(atoms))

x_gs = np.array(x_gs)

parameters = run_OLS_fit(x, y)
E_cand = np.matmul(x_gs, parameters)

gs_index = np.argmin(E_cand)
print(gs_index)
for i, row in enumerate(db_gs.select()):
    if i == gs_index:
        view(row.toatoms())

# %%

# Calculate lowest energy candidate for each sample,
# save lowest energy for each sample.

E_gs = []
gs_freq = np.zeros(x_gs.shape[0])

for i in range(len(flat_mcmc_samples[:, 0])):
    E = np.matmul(x_gs, flat_mcmc_samples[i, :33])
    index = np.argmin(E)
    E_gs.append(E[index])
    gs_freq[index] += 1

print(gs_freq/np.sum(gs_freq))
plt.hist(E_gs, bins=100)

gs_index_bayes = np.argmax(gs_freq)

for i, row in enumerate(db_gs.select()):
    if i == gs_index_bayes:
        view(row.toatoms())
print(gs_index_bayes)

# Frequency of being the lowest is converted to probability of the same.

# Plot the distribution of lowest energies.

# %%
plt.errorbar(range(33), np.mean(flat_mcmc_samples_copy[:, :33], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, :33], axis=0), fmt='.k', capsize=1.5, linewidth=0.7, markersize=4)
# plt.scatter(range(33),parameters)

# %%
cs.get_cluster_vector(atoms).shape

# %%
plt.errorbar(range(33), np.mean(flat_mcmc_samples_copy[:, :33], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, :33], axis=0), fmt='.k', capsize=1.5, linewidth=0.7, markersize=4)
plt.scatter(range(33), parameters)

# %%
plt.errorbar([0], np.mean(flat_mcmc_samples_copy[:, 0], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, 0], axis=0), fmt='.k', capsize=1.5, linewidth=0.7, markersize=4,label=r'$J_0$')
plt.errorbar([0], np.mean(flat_mcmc_samples_copy[:, 1], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, 1], axis=0), fmt='.C1', capsize=1.5, linewidth=0.7, markersize=4,label='Single atom cluster')

plt.errorbar(radii_2, np.mean(flat_mcmc_samples_copy[:, 2:21], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, 2:21], axis=0), fmt='.C2', capsize=1.5, linewidth=0.7, markersize=4,label='Pair clusters')
plt.errorbar(radii_3, np.mean(flat_mcmc_samples_copy[:, 21:33], axis=0), yerr=np.std(
    flat_mcmc_samples_copy[:, 21:33], axis=0), fmt='.C0', capsize=1.5, linewidth=0.7, markersize=4,label='Triplet clusters')
plt.legend()
plt.xlabel('Cluster radius [Å]')
plt.ylabel('ECI [meV]')
plt.tight_layout()
plt.savefig('ECIs.pdf')


# %%
plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.scatter(range(12),gs_freq/np.sum(gs_freq))
plt.xlabel('Ground state candidate [a.u.]')
plt.ylabel('Frequency of being ground state [a.u.]')
plt.tight_layout()
plt.savefig('gs_freq_mcmc.pdf')
plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.errorbar(range(12),np.mean(Es,axis=0),np.std(Es,axis=0), fmt='.C1', capsize=3, linewidth=1.5, markersize=12,label='MCMC')
plt.scatter(range(12),E_cand,label='OLS')
# plt.scatter(range(12),gs_freq/np.sum(gs_freq),'.k',)
plt.legend()
plt.xlabel('Ground state candidate [a.u.]')
plt.ylabel('Energy per atom [meV]')
plt.tight_layout()
plt.savefig('gs_Energy_OLS_v_MCMC.pdf')
 # %%
Es = []
E_gs = []

for i in range(len(flat_mcmc_samples[:, 0])):
    E = np.matmul(x_gs, flat_mcmc_samples[i, :33])
    Es.append(E)
Es = np.array(Es)
# %%


# %%
