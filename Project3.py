# %%
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
#import board
import numpy as np
from scipy.stats import norm, uniform,skew,kurtosis
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style("darkgrid")
sns.set_palette("deep")
sns.set(font='sans-serif')
%matplotlib inline
plt.rcParams['figure.dpi'] = 140
# %% Test
np.sum(board.experiment())
plt.plot(board.experiment())

# %%
mean = 0
n_runs = 100
for i in range(n_runs):
    test = board.experiment()
    mean += (np.sum([i*test[i]/1000 for i in range(32)]))
mean /= n_runs
print(mean)
plt.plot(test)
# %%


def controlled_experiment(alpha=0.25, s=0, size=1000):
    positions = np.zeros(size)  # Positions counted from the left most space
    prob_r = 0.5*np.ones(size)+s

    prev_right = np.random.rand(size) < prob_r
    positions += prev_right

    rows = 31
    for i in range(rows-1):
        prob_r = 0.5+(alpha*(prev_right-0.5)+s)
        prev_right = np.random.rand(size) < prob_r
        positions += prev_right

    counts = np.zeros(rows+1)
    for position in positions:
        counts[int(position)] += 1
    return counts


# %%
mean = []
variance = []

n_runs = 100
for j in range(n_runs):
    test = controlled_experiment(0.25, -0.25+0.5*j/n_runs)
    mean.append(np.sum([i*test[i]/1000 for i in range(32)]))
    variance.append(np.sum([(i-mean[j])**2*test[i]/1000 for i in range(32)]))

print(np.mean(mean))
plt.plot([-0.25+0.5*j/n_runs for j in range(n_runs)],
         variance, label='variance')
plt.plot([-0.25+0.5*j/n_runs for j in range(n_runs)], mean, label='mean')
plt.legend()
plt.xlabel('s')
# %%
mean = []
variance = []
for j in range(n_runs):
    test = controlled_experiment(j*0.5/n_runs, 0.2)
    mean.append(np.sum([i*test[i]/1000 for i in range(32)]))
    variance.append(np.sum([(i-mean[j])**2*test[i]/1000 for i in range(32)]))

print(np.mean(mean))
plt.plot([0.5*j/n_runs for j in range(n_runs)], variance, label='variance')
plt.plot([0.5*j/n_runs for j in range(n_runs)], mean, label='mean')
plt.legend()
plt.xlabel('alpha')
# %%

n_runs = 50
mean = np.zeros((n_runs, n_runs))
variance = np.zeros((n_runs, n_runs))
mode = np.zeros((n_runs, n_runs))
skewness = np.zeros((n_runs, n_runs))
for j in range(n_runs):
    for k in range(n_runs):
        test = controlled_experiment(0.5*j/n_runs,-0.25+0.5*k/n_runs)
        mean[j,k]=np.sum([i*test[i]/1000 for i in range(32)])
        variance[j,k]=np.sum([(i-mean[j,k])**2*test[i]/1000 for i in range(32)])
        mode[j,k]=(np.argmax(test))
        skewness[j,k]=np.sum([((i-mean[j,k])/np.sqrt(variance[j,k]))**3*test[i]/1000 for i in range(32)])
    #print(j)

plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.contourf([0.5*j/n_runs for j in range(n_runs)],
             [-0.25+0.5*j/n_runs for j in range(n_runs)], mean.T,20)
plt.ylabel('s [a.u.]')
plt.xlabel('alpha [a.u.]')
cb=plt.colorbar()
cb.set_label('Mean [a.u.]')
plt.tight_layout()
plt.savefig('dist_mean.pdf')

plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.contourf([0.5*j/n_runs for j in range(n_runs)],
             [-0.25+0.5*j/n_runs for j in range(n_runs)], variance.T,20)
plt.ylabel('s [a.u.]')
plt.xlabel('alpha [a.u.]')
cb=plt.colorbar()
cb.set_label('Variance [a.u.]')
plt.tight_layout()
plt.savefig('dist_variance.pdf')



plt.figure()
plt.contourf([0.5*j/n_runs for j in range(n_runs)],
             [-0.25+0.5*j/n_runs for j in range(n_runs)], (skewness).T,20)
plt.ylabel('s')
plt.xlabel('alpha')
plt.title('skew')
plt.colorbar()
# %% Helper functions to visualize the training of setup


def describe_topology(mlpr, verbose=False):
    w = []
    w.append('Perceptron topology:')
    w.append(f'  {"Input layer":20} - {mlpr.n_features_in_} neurons')
    hls = mlpr.hidden_layer_sizes
    if not isinstance(hls, tuple):
        hls = (hls, )
    for i, size in enumerate(hls):
        w.append(f'  {f"Hidden layer {i+1}":20} - {size} neurons  ')
    w.append(f'  {"Output layer":20} - {mlpr.n_outputs_} neurons')

    if verbose:
        print('\n'.join(w))

    return w


def train_test_score(mlpr, X_train, Y_train, X_test, Y_test, verbose=False):
    w = []
    train_s = mlpr.score(X_train, Y_train)
    test_s = mlpr.score(X_test, Y_test)
    w.append(f'{"Train score":20} {train_s:.5f}')
    w.append(f'{"Test score":20} {test_s:.5f}')

    if verbose:
        print('\n'.join(w))

    return w


def visualize_setup_1(mlpr, X_train, Y_train, X_test, Y_test):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    ax = axes[0]
    ax.plot(mlpr.loss_curve_)
    text = '\n'.join(describe_topology(mlpr, verbose=False) +
                     train_test_score(mlpr, X_train, Y_train, X_test, Y_test, verbose=False))
    ax.text(0.28, 0.98, text,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    ax = axes[1]
    x = np.linspace(0, 1, 200)
    y = mlpr.predict(x.reshape((-1, 1)))

    ax.plot(x, y, color='cornflowerblue', label='Predicted output')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.scatter(X_train, Y_train, 2, color='goldenrod', label='Training data')
    ax.scatter(X_test, Y_test, 2, color='seagreen', label='Testing data')

    ax.legend()

    fig.tight_layout()

# %%


hidden_layer_sizes = (16, 8)
reg_2 = MLPRegressor(hidden_layer_sizes, solver='sgd', activation='tanh',
                     tol=1e-6, n_iter_no_change=200, max_iter=2000,
                     alpha=0.0, momentum=0.9, learning_rate_init=0.1)

n_samples = 30000
n_bins = 32

X = np.zeros((n_samples, n_bins))
Y = np.zeros((n_samples, 2))

for i in range(n_samples):
    [alpha, s] = (np.random.rand(2)*[0.5, 0.5])+[0, -0.25]
    X[i, :] = controlled_experiment(alpha, s)*n_bins/1000
    Y[i, :] = [alpha, s]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.75)  # Split off 25% of data for test

reg_2.fit(X_train, Y_train)
print(
    f'Stopped after {reg_2.n_iter_} iterations with loss function {reg_2.loss_:.5f}')

fig, axes = plt.subplots(1, 1)

ax = axes
ax.plot(reg_2.loss_curve_)

text = '\n'.join(describe_topology(reg_2, verbose=False) +
                 train_test_score(reg_2, X_train, Y_train, X_test, Y_test, verbose=False))
ax.text(0.28, 0.98, text,
        fontsize=8, ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
plt.tight_layout()
plt.savefig('NN_training.pdf')
# %% mean squared errors for test set
MSE = np.sum((reg_2.predict(X_train)-Y_train)
                     ** 2, axis=0)/Y_train.shape[0]
print('MSE alpha ', MSE[0])
print('MSE s ', MSE[1])

#%% MSE for a grid of parameters

n_runs = 20
MSE_grid = np.zeros((n_runs,n_runs,2))
Y = np.zeros((n_runs,n_runs, 2))
n_samples = 20
X = np.zeros((n_samples, n_bins))
alpha = np.linspace(0,0.5,n_runs)
s = np.linspace(-0.25,0.25,n_runs)
for j in range(n_runs):
    for k in range(n_runs):
        Y[j,k, :] = [alpha[j], s[k]]
        for i in range(n_samples):     
            X[i, :] = controlled_experiment(alpha[j], s[k])*n_bins/1000
        MSE_grid[j,k,:] = np.sum((reg_2.predict(X)-Y[i,j, :]) ** 2, axis=0)/n_samples

plt.contourf(alpha,s,MSE_grid[:,:,0].T)
plt.title('MSE alpha')
plt.xlabel('alpha')
plt.ylabel('s')
plt.colorbar()
plt.figure()
plt.contourf(alpha,s,MSE_grid[:,:,1].T)
plt.title('MSE s')
plt.xlabel('alpha')
plt.ylabel('s')
plt.colorbar()
# %% ML-supported ABC algorithm


def ABC(y_obs, kernel, NN, h_scale = 1,n_samples = 100,max_runs = 10000):
    theta_m = NN.predict((y_obs,))[0]

    stat_obs = statistic(y_obs)
    h = np.array(statistic_std(theta_m))*h_scale
    # K0=kernel(np.array([0,0]),h)

    thetas = np.empty((n_samples, 2))
    i = 0
    j = 0
    while i < n_samples:
        theta = NN.predict((controlled_experiment(
            theta_m[0], theta_m[1])*n_bins/1000,))[0]

        y = controlled_experiment(theta[0], theta[1])
        stat_test = statistic(y)
        stat_test-stat_obs
        acc_prob = kernel(stat_test-stat_obs, h)  # /K0

        if(np.random.rand() < acc_prob):
            thetas[i, :] = theta
            i += 1

        j += 1
        if (j > max_runs):
            print('Acceptance ratio ', i/j)
            raise Exception(
                "Max number of iterations reached! Acceptance rate too low, increase h.")
    print('Number of iterations ', j)
    print('Acceptance ratio ', i/j)
    return thetas


def ABC_latent_var_elim(y_obs, kernel, NN, h_scale = 1, n_samples = 100, max_runs = 10000):
    theta_m = NN.predict((y_obs,))[0]

    stat_obs = statistic(y_obs)
    h = np.array(statistic_std(theta_m))*h_scale
    # K0=kernel(np.array([0,0]),h)

    g_s = norm(loc=theta_m[1],scale=np.sqrt(MSE[1]))
    g_alpha=norm(loc=theta_m[0],scale=np.sqrt(MSE[0])) #uniform(0,0.5)

    thetas = np.empty((n_samples, 2))
    i = 0
    j = 0
    while i < n_samples:
        s = g_s.rvs()
        alpha=g_alpha.rvs()
        
        y = controlled_experiment(alpha, s)
        stat_test = statistic(y)
        stat_test-stat_obs
        acc_prob = kernel(stat_test-stat_obs, h)/g_alpha.pdf(alpha)  # /K0

        if(np.random.rand() < acc_prob):
            thetas[i, :] = [alpha,s]
            i += 1

        j += 1
        if (j > max_runs):
            print('Acceptance ratio ', i/j)
            raise Exception(
                "Max number of iterations reached! Acceptance rate too low, increase h.")
    print('Number of iterations ', j)
    print('Acceptance ratio ', i/j)
    return thetas

def ABC_importance_sampling(y_obs, kernel, NN, h_scale = 1, max_weight_sum = 100, max_runs = 10000):
    theta_m = NN.predict((y_obs,))[0]

    if not( 0<= theta_m[0] <=.5 ):
        print("Alpha outside bounds at alpha=",theta_m[0])
    if not(-0.25<= theta_m[1] <=0.25 ):
        print("S outside bounds at s=",theta_m[1])
    stat_obs = statistic(y_obs)
    h = np.array(statistic_std(theta_m))*h_scale
    # K0=kernel(np.array([0,0]),h)

    g_s = norm(loc=theta_m[1],scale=np.sqrt(MSE[1]))
    g_alpha=norm(loc=theta_m[0],scale=np.sqrt(MSE[0]))#uniform(0,0.5)


    thetas = np.empty((max_runs, 2))
    weights = np.empty(max_runs)
    weight_sum = 0
    i = 0
    while weight_sum < max_weight_sum:
        s = g_s.rvs()

        if not(-0.25 <= s <= 0.25):
            continue

        alpha=g_alpha.rvs()
        
        y = controlled_experiment(alpha, s)
        stat_test = statistic(y)
        stat_test-stat_obs
        weight = kernel(stat_test-stat_obs, h)/g_s.pdf(s)/g_alpha.pdf(alpha)  # /K0
        weight_sum+=weight
        weights[i] = weight
        
        thetas[i, :] = [alpha,s]
        
        i += 1
        if(i%1000==0):
            print("weight_sum = ",weight_sum," for i=",i)
        if (i > max_runs-1):
            break
            # print('Weight_sum ', weight_sum)
            # raise Exception("Max number of iterations reached! Increase h.")
    
    print('Number of iterations ', i)
    print('Weight_sum ', weight_sum)
    return thetas[:i,:],weights[:i]

def kernel_gaussian(u, h):
    return np.exp(-np.sum((u/h)**2))


# us = np.linspace(-2, 2)
# K = np.empty(50)
# for i in range(50):
#     K[i] = kernel_gaussian(us[i], np.array(1))
# plt.plot(us, K)


def statistic(y):
    mean = np.sum([i*y[i] for i in range(len(y))])/np.sum(y)
    var = np.sum([y[i]*(i-mean)**2 for i in range(len(y))])/np.sum(y)
    return np.array([mean, var])


def statistic_std(theta, n_runs=500):
    stat = np.empty((n_runs, 2))
    for i in range(n_runs):
        stat[i, :] = statistic(controlled_experiment(theta[0], theta[1]))
    std = np.std(stat, axis=0)
    std_of_mean = std[0]
    std_of_var = std[1]
    return std_of_mean, std_of_var

#%%
def weighted_mean(samples,weights):
    return np.sum(samples*weights)/np.sum(weights)

def weighted_std(samples, weights):
    mean=weighted_mean(samples,weights)
    return np.sqrt(weighted_mean((samples-mean)**2, weights))

# %% Test the method with controll values for alpha and s

alpha = 0.25
#s = -0.2

n_samples=100
n_random_runs=50
theta_samples_var_elim=np.empty((n_samples*n_random_runs,2))

for i in range(n_random_runs):
    s = np.random.rand()*0.5-0.25
    y_obs_test=controlled_experiment(alpha, s)*n_bins/1000
    theta_samples_var_elim[i*n_samples:(i+1)*n_samples,:] =ABC_latent_var_elim(y_obs_test, kernel_gaussian, reg_2,h_scale=1,n_samples=n_samples,max_runs = 3000*n_samples)
    

plt.hist2d(theta_samples_var_elim[:, 0], theta_samples_var_elim[:, 1],bins=30)
plt.scatter(alpha,s)
plt.figure()
n, bins, patches = plt.hist(theta_samples_var_elim[:, 0],bins=50)
plt.plot([alpha, alpha], [0, np.max(n)], label='true value')
plt.legend()

#%%
alpha = 0.25
s = -0.2

s_samples = np.empty((0))
max_weight_sum=1
h_scale=1
n_random_runs=200

theta_samples=np.empty((0,2))
weights=np.empty(0)

running_mean=[]
running_std=[]

for i in range(n_random_runs):
    s = np.random.rand()*0.5-0.25
    y_obs_test=controlled_experiment(alpha, s)*n_bins/1000
    theta_samples_tmp, weight_tmp = ABC_importance_sampling(y_obs_test, 
        kernel_gaussian, reg_2,h_scale=h_scale,max_weight_sum=max_weight_sum,
        max_runs = 10000*100)
    theta_samples = np.append(theta_samples,theta_samples_tmp,axis=0)
    weights = np.append(weights,weight_tmp)
    s_samples = np.append(s_samples,s)
    running_mean.append(weighted_mean(theta_samples[:,0],weights))
    running_std.append(weighted_std(theta_samples[:,0],weights))

#%%

plt.hist2d(theta_samples[:, 0], theta_samples[:, 1],weights=weights,bins=30)
plt.scatter((alpha,)*len(s_samples),s_samples)
plt.xlabel
plt.figure()
n, bins, patches = plt.hist(theta_samples[:, 0],weights=weights,bins=50,density=True)
plt.plot([alpha, alpha], [0, np.max(n)], label='true value')
plt.legend()
plt.xlabel(r'$\alpha [a.u.]$')
plt.ylabel('Probability density [a.u.]')
plt.tight_layout()
plt.savefig('control_dist.pdf')

#%%

plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.plot(running_mean)
plt.xlabel('Number of experiments run')
plt.ylabel(r'Mean $\alpha$ prediction [a.u.]')
plt.tight_layout()
plt.savefig('mean_conv.pdf')
plt.figure(figsize=(6.4*0.8, 4.8*0.8))
plt.plot(np.array(running_std))
plt.xlabel('Number of experiments run')
plt.ylabel(r'Std of $\alpha$ prediction [a.u.]')
plt.tight_layout()
plt.savefig('std_conv.pdf')

# %%
theta_samples=ABC(controlled_experiment(0,0),kernel_gaussian,reg_2)
#%%
#plt.hist2d(theta_samples[:,0],theta_samples[:,1])

running_mean=[]
running_std=[]

for i in range(n_random_runs):
    running_mean.append(np.mean(theta_samples[:(i+1)*n_samples,0]))
    running_std.append(np.std(theta_samples[:(i+1)*n_samples,0]))

plt.plot(running_mean)
plt.title(r'Running mean($\alpha$)')
plt.figure()
plt.plot(running_std)
plt.title(r'Running std($\alpha$)')

#%%
alpha = 0.15
s = 0.1
y_obs_test=controlled_experiment(alpha, s)*n_bins/1000
#%%
theta_samples,weights = ABC_importance_sampling( y_obs_test, kernel_gaussian, reg_2,h_scale=2,max_weight_sum=20,max_runs = 10000*100)

plt.hist2d(theta_samples[:, 0], theta_samples[:, 1],weights=weights,bins=30)
plt.scatter(alpha,s)
plt.figure()
n, bins, patches = plt.hist(theta_samples[:, 0],weights=weights,bins=50)
plt.plot([alpha, alpha], [0, np.max(n)], label='true value')
plt.legend()

#%% plot g(theta)
plt.hist2d(theta_samples[:, 0], theta_samples[:, 1],bins=30)
plt.scatter(alpha,s)
plt.figure()
n, bins, patches = plt.hist(theta_samples[:, 0],bins=50)
plt.plot([alpha, alpha], [0, np.max(n)], label='true value')
plt.legend()
#%%
theta_samples = ABC_latent_var_elim( y_obs_test, kernel_gaussian, reg_2,h_scale=0.5,n_samples=100,max_runs = 1000*100)

plt.figure()
plt.hist2d(theta_samples[:, 0], theta_samples[:, 1])
plt.scatter(alpha,s)
plt.figure()
n, bins, patches = plt.hist(theta_samples[:, 0])
plt.plot([alpha, alpha], [0, np.max(n)], label='true value')
plt.legend()
#%%

theta_test=np.empty((1000,2))
alpha,s=0.2,-0.25
for i in range(1000):
    theta_test[i,:] = reg_2.predict((controlled_experiment(alpha,s)*n_bins/1000,))[0]

plt.hist2d(theta_test[:,0],theta_test[:,1],bins=20)
plt.scatter(alpha,s)
plt.xlabel('alpha')
plt.ylabel('s')

#%%


def binary_search(arr, x):
    low = 0
    high = len(arr)-1

    mid = (high + low) // 2
    # If element is present at the middle itself
    if arr[mid] == x:
        return mid

    # If element is smaller than mid, then it can only
    # be present in left subarray
    elif arr[mid] > x:
        return binary_search(arr[:mid], x)

    # Else the element can only be present in right subarray
    else:
        return mid+binary_search(arr[mid:], x)


# Test array
arr = [2, 3, 4, 10, 40]
x = 3.5

# Function call
binary_search(arr, x)

# %%

grid = 20
alpha = np.linspace(0, 0.5, grid)
s = np.linspace(-0.25, 0.25, grid)

n_runs = 100
stat = np.empty((grid, grid, n_runs, 2))
for i in range(grid):
    for j in range(grid):
        for k in range(n_runs):
            stat[i, j, k] = statistic(controlled_experiment(alpha[i], s[j]))

std = np.std(stat, axis=2)
std_of_mean = std[:, :, 0]
std_of_var = std[:, :, 1]

plt.contourf(alpha, s, std_of_mean.T)
plt.ylabel('s')
plt.xlabel('alpha')
plt.colorbar()
plt.title('standard deviation o the mean')
plt.figure()

plt.contourf(alpha, s, std_of_var.T)
plt.ylabel('s')
plt.xlabel('alpha')
plt.title('standard deviation of the variance')
plt.colorbar()

# %%



