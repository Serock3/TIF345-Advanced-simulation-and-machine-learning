#%%
import board
import numpy as np
from matplotlib import pyplot as plt

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

def controlled_experiment(alpha=0.25,s=0,size=1000):
    positions = np.zeros(size) # Positions counted from the left most space
    prob_r=0.5*np.ones(size)+s

    prev_right=np.random.rand(size) < prob_r
    positions+=prev_right
    
    rows = 31
    for i in range(rows-1):
        prob_r=0.5+(alpha*(prev_right-0.5)+s)
        prev_right=np.random.rand(size) < prob_r
        positions+=prev_right

    counts = np.zeros(rows+1)
    for position in positions:
        counts[int(position)]+=1
    return counts
# %%
mean = 0
n_runs = 100
for i in range(n_runs):
    test = controlled_experiment()
    mean += (np.sum([i*test[i]/1000 for i in range(32)]))
mean /= n_runs
print(mean)
plt.plot(test)