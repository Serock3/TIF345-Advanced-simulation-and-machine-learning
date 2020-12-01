# %% Import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gamma, invgamma, t, norm, norminvgauss, mode, multivariate_normal
import corner
import emcee
import statistics
sns.set_context("paper", font_scale=1.5)
sns.set_style("darkgrid")
sns.set_palette("deep")
sns.set(font='sans-serif')
# %matplotlib inline
plt.rcParams['figure.dpi'] = 140

np.random.seed(123)

#%% Import data

data = pd.read_csv("SCPUnion2.1_mu_vs_z.txt", skiprows = 5, delimiter='\t',names=['Name','Redshift','Distance Modulus','Distance modulus error','p'])

z_meas = data['Redshift']
mu_meas = data['Distance Modulus']
sig_meas = data['Distance modulus error']
w=1/sig_meas**2/np.sum(1/sig_meas**2)
low_lim = z_meas<0.5 # pick out low limit z


# %% define pdfs and helper funcs
c = 299792.458

def mean_mode_2_IG_alpha_beta(mean,mode):
    alpha = (mode + mean)/(mean-mode)
    beta = (2 * mode * mean)/(mean-mode)
    return alpha, beta
a0,b0 = mean_mode_2_IG_alpha_beta(1,0.2)

def dL_eq17(z,H0,q0):
    return c/H0*(z+0.5*(1-q0)*z**2)

def mu_eq4(z,H0,q0):
    return 5*np.log10(dL_eq17(z,H0,q0))+25

def log_likelihood(par, z, y, w):
    H0,q0,sig2=par
    model = mu_eq4(z,H0,q0)
    n = len(y)
    return -0.5*(np.sum((y-model)**2*n*w/sig2)) - (n/2)*np.log(sig2) #+ np.sum(np.log(np.sqrt(w)))

def log_sig2_prior(par):
    H0,q0,sig2=par
    
    lp = invgamma.logpdf(sig2,a=a0, scale=b0)
    return lp

def log_posterior(par,z,y,w):
    lp = log_sig2_prior(par)
    #lp = np.where(np.isnan(lp),-np.inf,lp)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(par, z, y, w)

    return ll + lp

# %%
ndim, nwalkers = 3, 40
start_pos = [70,0,0.1] + [1e-2,1e-2,1e-5]*np.random.randn(nwalkers, ndim)

# %%

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(z_meas[low_lim], mu_meas[low_lim],w[low_lim]))
sampler.run_mcmc(start_pos, 4000, progress=True)

# %%

def simple_mcmc_analysis(sampler, par, label, burn_in, chain_from_file=False):

    if not chain_from_file:
        print(f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):0.3f}')
    # discard the first 'burn_in' samples 
    
    # thinning means that you only keep every nth sample. E.g. thinning=10 -> keep every 10th sample.
    # This can be useful for reducing long autocorrelation lenghts in a chain. However, thinning is expensive.
    # A thinned chain must be run E.g. 10x longer to reach the desired number of samples.
    # One can argue that thinning is not an advantageous strategy. So keep thinning = 1
    thinning = 1
    flat_mcmc_samples = sampler.get_chain(discard=burn_in,thin=thinning, flat=True)
    print(f'Discarding {nwalkers*burn_in} steps as burn-in')
    print(f'Chain length:{len(flat_mcmc_samples)}')

    fig1 = plt.figure()
    plt.plot(flat_mcmc_samples[:,par],color='gray',alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel(label)
    plt.xlim(0,len(flat_mcmc_samples))

    return flat_mcmc_samples

flat_mcmc_samples = simple_mcmc_analysis(sampler, par=0, label=f'$H_0$', burn_in=100)

fig = corner.corner(flat_mcmc_samples[:,:2],labels=[r"$H_0$", r"$q_0$", r"$\sigma^2$"],show_titles=True)

plt.savefig('H0q0_post.pdf')


# %% Checking H0

size = 50
z_lim = np.linspace(0,0.5,size)
av = np.empty(size)

for i in range(size):
    low_lim = z_meas<z_lim[i] # pick out low limit z
    H0=c*z_meas[low_lim]*10**((25-mu_meas[low_lim])/5)
    av[i] = np.mean(H0)
# plt.hist(H0,bins=15)
# print("Mean: ",np.mean(H0))
plt.plot(z_lim,av)
plt.xlabel('Upper limit for $z$ data')
plt.ylabel('$H_0$')
plt.savefig('H0smallz.pdf')
# %% Plot
x0 = np.linspace(np.min(z_meas),np.max(z_meas),50)
inds = np.random.randint(len(flat_mcmc_samples), size=100)
for ind in inds:
    sample = flat_mcmc_samples[ind,:]
    plt.plot(x0, mu_eq4(x0,sample[0],sample[1]), "C1", alpha=0.1)
plt.errorbar(z_meas,mu_meas , yerr=sig_meas, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
# plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

# %%
partial_size=1000
z0 = np.linspace(np.min(z_meas),np.max(z_meas),partial_size)
inds = np.random.randint(len(flat_mcmc_samples), size=partial_size)

y_pred_mean=np.zeros(partial_size)
y_pred_std_high=np.zeros(partial_size)
y_pred_std_low=np.zeros(partial_size)

for i,ind in enumerate(inds):
    y_pred=mu_eq4(z0[i],flat_mcmc_samples[inds,0],flat_mcmc_samples[inds,1])
    y_pred_mean[i]=np.mean(y_pred)
    # y_pred_std_high[i]=y_pred_mean[i]+1.96 *np.std(y_pred)
    # y_pred_std_low[i]=y_pred_mean[i]-1.96*np.std(y_pred)
    quantiles = statistics.quantiles(y_pred,n=200)
    y_pred_std_high[i] = quantiles[-1]
    y_pred_std_low[i]= quantiles[0]
    
partial_size=len(z_meas)
inds_data = np.random.randint(len(z_meas), size=partial_size)
plt.errorbar(z_meas[inds_data],mu_meas[inds_data] , yerr=sig_meas[inds_data], fmt=".k", capsize=0, linewidth=1, markersize=4,alpha=0.3)

plt.plot(z0,y_pred_mean,label='Mean prediction', linewidth=1)
col = plt.gca().lines[-1].get_color()
plt.fill_between(z0,y_pred_std_low,y_pred_std_high,alpha=0.3,color=col,label='99% BCI')
plt.ylabel('$\mu$')
plt.xlabel('$z$')

plt.legend()
plt.savefig('predictive_post.pdf')

# %% How could one improve the inference?

# More data at higher Z, more terms in eq 17


# %% Task 2
import scipy.integrate as integrate

def E_LCDM(z,Omega_M0):
    return Omega_M0*((1+z)**3-1)+1

def E_wCDM(z,pars):
    Omega_M0,w = pars
    return Omega_M0*(1+z)**3+(1-Omega_M0)*(1+z)**(3*(1+w))

def dL_eq15(z,E,E_pars,H0):
    steps = 100
    # z0 = np.linspace(0,z,steps)
    z0 = np.linspace((0)*len(z),z,steps)
    return c*(1+z)/H0*np.trapz(1/np.sqrt(E(z0,E_pars)),z0,axis=0)

def mu_eq4(z,H0, E, E_pars):
    return 5*np.log10(dL_eq15(z,E,E_pars,H0))+25

def log_likelihood(E_pars, E, H0, z, y, w):
    if len(E_pars)>1:
        if E_pars[0]<0 or E_pars[0]>1:
            return -np.inf
    else:
        if E_pars<0 or E_pars>1:
            return -np.inf

    model = mu_eq4(z,H0, E, E_pars)
    n = len(y)
    return -0.5*np.sum((y-model)**2*n*w)# - (n/2)*np.log(sig2) #+ np.sum(np.log(np.sqrt(w)))

def neg_log_likelihood(E_pars, E, H0, z, y, w):
    return -1*log_likelihood(E_pars, E, H0, z, y, w)

# def dL_eq15int(z,E,E_pars,H0):
#     return c*(1+z)/H0*integrate.quad(lambda z0: 1/np.sqrt(E(z0,E_pars)),0,z)[0]

# def dL_eq15intAll(z,E,E_pars,H0):
#     result = np.empty(len(z))
#     for i in range(len(z)):
#         result[i]= dL_eq15int(z[i],E,E_pars,H0)
#     return result
# %%
import scipy.optimize as opt

Omega_M0=0.5
H0=70
w0=-1

LCDM=opt.minimize(neg_log_likelihood, Omega_M0, args=(E_LCDM, H0, z_meas, mu_meas, w))
# wCDM=opt.minimize(neg_log_likelihood, np.array([Omega_M0,w0]), args=(E_wCDM, H0, z_meas, mu_meas, w))
wCDM=opt.differential_evolution(neg_log_likelihood, [(0,1),(-4,2)], args=(E_wCDM, H0, z_meas, mu_meas, w))

mll_LCDM= -1*LCDM.fun
mll_wCDM= -1*wCDM.fun

Omega_M_LCDM=LCDM.x
Omega_M_wCDM, w_wCDM = wCDM.x

n=len(z_meas)


BIC_LCDM=2*mll_LCDM-1*np.log(n)
BIC_wCDM=2*mll_wCDM-2*np.log(n)
AIC_LCDM=2*mll_LCDM-2*1
AIC_wCDM=2*mll_wCDM-2*2
print('BIC_LCDM ',BIC_LCDM)
print('BIC_wCDM ',BIC_wCDM)
print('AIC_LCDM ',AIC_LCDM)
print('AIC_wCDM ',AIC_wCDM)

print('Omega_M_LCDM ',Omega_M_LCDM)
print('Omega_M_wCDM ',Omega_M_wCDM)
print('w_wCDM',w_wCDM)

#%% Redefining the measurement variance
def log_likelihood(E_pars, E, H0, z, y, w):
    if len(E_pars)>1:
        if E_pars[0]<0 or E_pars[0]>1:
            return -np.inf
    else:
        if E_pars<0 or E_pars>1:
            return -np.inf

    model = mu_eq4(z,H0, E, E_pars)
    n = len(y)
    return -0.5*np.sum((y-model)**2/sig_meas**2)# - (n/2)*np.log(sig2) #+ np.sum(np.log(np.sqrt(w)))

# %%

start_pos = [0.5] + [1e-1]*np.random.randn(nwalkers, 1)

sampler = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(E_LCDM, H0,z_meas, mu_meas,w))
sampler.run_mcmc(start_pos, 2000, progress=True)


#%%
flat_mcmc_samples = simple_mcmc_analysis(sampler, par=0, label=f'$H_0$', burn_in=100)

fig = corner.corner(flat_mcmc_samples[:,:2],labels=[r"$\Omega_{M,0}$", r"$q_0$", r"$\sigma^2$"],show_titles=True)

plt.savefig('Omega_M0.pdf')
# %%

size = 30
x=np.linspace(-4,2,size)
y=np.zeros(size)
for i in range(size):
    y[i]=log_likelihood([0.21,x[i]],E_wCDM, H0,z_meas, mu_meas,w)

plt.plot(x,y)


# %%
a0,b0 = mean_mode_2_IG_alpha_beta(1,0.25)
size = 100
x=np.linspace(0,2,size)
y=np.zeros(size)
for i in range(size):
    y[i]=invgamma.pdf(x[i],a=a0, scale=b0)

plt.plot(x,y)

# %%


