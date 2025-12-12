#!/usr/bin/env python
# coding: utf-8

# ## Multi-fidelity Bayesian optimization (MFBO) of COFs for Xe/Kr separations.
# 1. We have a set of COFs from a database. Each COF is characterized by a feature vector $$x \in \mathcal{X} \subset R^d$$ were d=14.
# 
# 2. We have **two different types** of simulations to calculate **the same material property**, the adsorptive Xe/Kr selectivity $S_{Xe/Kr}$. However, we only have a single objective: to maximize the high-fidelity selectivity. 
# $$\arg\max_{x \in \mathcal{X}}[S^{(\ell=\text{high})}_{Xe/Kr}(x)]$$
# 
# 3. Multi-Fidelity options: 
#     1. low-fidelity  => Henry coefficient calculation - MC integration: $S_{Xe/Kr}^{\text{low}} = \frac{H_{Xe}}{H_{Kr}}$
#     2. high-fidelity => GCMC mixture simulation - 80:20 (Kr:Xe) at 298 K and 1.0 bar: $S_{Xe/Kr}^{\text{high}} = \frac{n_{Xe} / n_{Kr}}{y_{Xe}/y_{Kr}}$
# 
# 
# 3. We will initialize the surrogate model with a few (3) COFs with simulations under **both** fidelities.
#     1. The fist COF will be the one closest to the center of the normalized feature space
#     2. The rest will be chosen to maximize diversity of the training set
# 
# 
# 4. Model:
#     1. Botorch GP surrogate model: [SingleTaskMultiFidelityGP](https://botorch.org/api/models.html#module-botorch.models.gp_regression_fidelity) or [FixedNoiseMultiFidelityGP](https://botorch.org/api/models.html#botorch.models.gp_regression_fidelity.FixedNoiseMultiFidelityGP)
#         - Needed to use [this](https://botorch.org/api/optim.html#module-botorch.optim.fit) optimizer to correct matrix jitter
#     2. We  use the augmented-EI (aEI) acquisition function from [here](https://link.springer.com/content/pdf/10.1007/s00158-005-0587-0.pdf)
# 
# 
# -  Helpful [tutorial](https://botorch.org/tutorials/discrete_multi_fidelity_bo) for a similar BoTorch Model used

# In[ ]:


#!pip install gpytorch


# In[ ]:


#!pip install botorch 


# In[ ]:


import torch
import gpytorch

from botorch.models import SingleTaskMultiFidelityGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.analytic import PosteriorMean

from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
#from botorch import fit_gpytorch_model 
from botorch.fit import fit_gpytorch_mll
#from botorch.optim.fit import fit_gpytorch_model
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import norm
import numpy as np
import pickle
import h5py # for .jld2 files
import os
import time

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error


# In[ ]:


###
#  figure settings 
###
import seaborn as sns
sns.set(style='ticks', palette='Set2', font_scale=1.5, rc={"lines.linewidth": 3})
sns.despine()
# plt.rcParams.update({'font.size': 16})
# plt.rcParams['figure.dpi'] = 1200 # 600-1200 for paper quality

save_figures = True


# In[ ]:


discrete_fidelities = [1/3, 2/3] # set of discrete fidelities (in ascending order) to select from this cannot be changed.


# ## Load Data

# In[ ]:


ablation_study_flag = False # make features have no information for a baseline. to gauge feature importance


# first, the targets (simulated adsorption) and features of the COFs.

# In[ ]:


file = h5py.File("./targets_and_raw_features.jld2", "r")

xtal_names = file['COFs'][:]

feature_names = file['feature_names'][:]
feature_names = [fn.decode() for fn in feature_names]

# feature matrix
X = torch.from_numpy(np.transpose(file["X"][:]))

if ablation_study_flag:
    # shuffle columns
    for j in range(X.size()[1]):
        shuffled_row_ids = torch.randperm(X.size()[0])
        X[:, j] = X[shuffled_row_ids, j]

# simulation data
y = [torch.from_numpy(np.transpose(file["henry_y"][:])), 
     torch.from_numpy(np.transpose(file["gcmc_y"][:]))]  

print("top COF = ", xtal_names[np.argmax(y[1])])
# associated simulation costs
cost = [np.transpose(file["henry_total_elapsed_time"][:]), # [min]
        np.transpose(file["gcmc_elapsed_time"][:])]        # [min]

# total number of COFs in data set
nb_COFs = X.shape[0]


# second, the COFs to initialize the surrogate model.

# In[ ]:


init_cof_ids_file = pickle.load(open('../search_results/initializing_cof_ids_normalized.pkl', 'rb'))

init_cof_ids = init_cof_ids_file['init_cof_ids']

# total number of BO searches to run = number of initializing sets
nb_runs = len(init_cof_ids)
if ablation_study_flag:
    nb_runs = 10


# In[ ]:


import pickle

with open('../search_results/initializing_cof_ids_normalized.pkl', 'rb') as f:
    init_file = pickle.load(f)

print(init_file.keys())


# In[ ]:


init_cof_ids = init_file['init_cof_ids']
print(type(init_cof_ids))
print(len(init_cof_ids))


# In[ ]:


for i, init_set in enumerate(init_cof_ids):
    print(f"Initial training set for run {i}:")
    print(init_set)
    print()


# some tests on the input data.

# In[ ]:


for i, f in enumerate(feature_names):
    print("{}: {}".format(i, f))


# In[ ]:


rnd_COF = b'19060N2_ddec.cif' # random COF

id_rnd_COF = np.where(xtal_names == rnd_COF)[0]

# does the low-fidelity selectivity match that manually read from the simulation output file?
assert np.isclose(y[0][id_rnd_COF].item(), 722.409 / 202.085)

# does the high-fidelity selectivity match that manually read from the simulation output file?
assert np.isclose(y[1][id_rnd_COF].item(), (6.1558810248879325 / 6.842906773660418) / (20/80))

# manually check some features
if not ablation_study_flag:
    # ASA_m^2/g = 4363.81 in Zeo++ output file. this better match!
    assert np.isclose(X[id_rnd_COF, 2].item(), 4363.81)

    # mol fraction of N = 0.04807692307692308 in the xtal. this better match!
    assert np.isclose(X[id_rnd_COF, 9].item(), 0.04807692307692308)
    
    # sum of mol frac's = 1
    assert X[id_rnd_COF, 4:].sum().item() == 1
    
    # pore diameter = 15.12574 from Zeo++ output file
    assert X[id_rnd_COF, 0] == 15.12574
    
    # void fraction = 0.58554 from Zeo++ output file.
    assert X[id_rnd_COF, 1] == 0.58554
    
    # xtal density from Zeo++ output = 0.604869
    assert np.isclose(X[id_rnd_COF, 3], 0.604869 * 1000, atol=0.1) # unit convert cuz computed in PM.jl


# now that we've tested, normalize the features to lie in [0, 1]

# In[ ]:


for j in range(X.size()[1]):
    X[:, j] = (X[:, j] - torch.min(X[:, j]).item()) / (torch.max(X[:, j]).item() - torch.min(X[:, j]).item())


# In[ ]:


# normalization worked
assert np.allclose(torch.min(X, 0).values, torch.zeros(14))
assert np.allclose(torch.max(X, 0).values, torch.ones(14))


# In[ ]:


# assert first COF closest to mean
if not ablation_study_flag:
    x_mean = torch.mean(X, 0)
    assert init_cof_ids[0][0] == np.argmin([np.linalg.norm(x_mean - X[j, :].detach().numpy()) for j in range(nb_COFs)])
    # assert next COF furthest.
    assert init_cof_ids[0][1] == np.argmax([np.linalg.norm(X[init_cof_ids[0][0], :] - X[j, :].detach().numpy()) for j in range(nb_COFs)])


# print stuff

# In[ ]:


# cost
print("total high-fidelity cost:", sum(cost[1]).item(), "[min]")
print("total low-fidelity cost: ", sum(cost[0]).item(), "[min]")
print("average high-fidelity cost:", np.mean(cost[1]), "[min]")
print("average low-fidelity cost: ", np.mean(cost[0]), "[min]")
print("average cost ratio:\t   ", np.mean(cost[1] / cost[0]))

# data shape
print("\nraw data - \n\tX:", X.shape)
for f in range(2):
    print("\tfidelity:", f)
    print("\t\ty:", y[f].shape)
    print("\t\tcost: ", cost[f].shape)
    
# normalization check
print("\nEnsure features are normalized - ")
print("max:\n", torch.max(X, 0).values)
print("min:\n", torch.min(X, 0).values)
print("width:\n",torch.max(X, 0).values - torch.min(X, 0).values)
print("mean:\n", torch.mean(X, 0))
print("std:\n", torch.std(X, 0))


# ## Helper Functions

# #### Post-Search Analysis

# In[ ]:


# return list of fidelity id's (0's and 1's) from the acquired set.
def get_f_ids(acquired_set):
    if acquired_set.dim() == 0:
        return acquired_set.round().to(dtype=int)
    else: 
        f_ids = [a[0].round().to(dtype=int) for a in acquired_set]
        return torch.tensor(f_ids)


# In[ ]:


# get the list of high-fidelity y_max's from iter-to-iter
# returns an array.
# element i is best y-max high fidelity seen up to iteration i.
def get_y_maxes_hf_acquired(acquired_set):    
    nb_iters = len(acquired_set)
    y_maxes = np.zeros(nb_iters)
    # we want the maximum y value (only high-fidelity) up to a given iteration
    y_max = 0.0 # update this each iteration.
    for i, (f_val, cof_id) in enumerate(acquired_set):
        f_id = get_f_ids(torch.tensor(f_val))
        assert f_id in [0, 1]
        y_acq_this_iter = y[f_id][int(cof_id)]
        # i is iteration index
        if f_id == 1 and y_acq_this_iter > y_max:  
            y_max = y_acq_this_iter # over-write max
        y_maxes[i] = y_max 
    return y_maxes


# In[ ]:


# find accumulated cost, given acquired set.
# returns an array.
# element i is cost accumulated till iteration i
def accumulated_cost(acquired_set):
    nb_iters = len(acquired_set)
    accumulated_cost = np.zeros(nb_iters)
    for i, (f_val, cof_id) in enumerate(acquired_set):
        cof_id = int(cof_id.item())
        f_id = f_val.round().to(dtype=int).item()
        if i == 0:
            accumulated_cost[i] = cost[f_id][cof_id]
        else:
            accumulated_cost[i] = accumulated_cost[i-1] + cost[f_id][cof_id]
    return accumulated_cost


# In[ ]:


# calcualte the fraction of sims up to that point that are a given fidelity
# entry i is fraction of sims up to that point that are fidelity fidelity.
def calc_fidelity_fraction(acquired_set, fidelity):
    assert fidelity in [1/3, 2/3] 
    nb_iters = len(acquired_set)
    fid_frac = np.zeros(nb_iters)
    for i in range(nb_iters):
        fid_frac[i] = sum(acquired_set[:, 0][:i+1] == fidelity) / (i+1)
    return fid_frac


# #### constructing initial acquired set

# In[ ]:


def initialize_acquired_set(initializing_COFs, discrete_fidelities):
    return torch.tensor([[f_id, cof_id] for cof_id in initializing_COFs for f_id in discrete_fidelities])


# #### building the inputs for the surrogate model

# In[ ]:


# construct feature matrix of acquired points.
# the last entry is the fidelity parameter.
def build_X_train(acquired_set):
    cof_ids = [a[1].to(dtype=int) for a in acquired_set]
    f_ids = torch.tensor([a[0] for a in acquired_set])
    assert f_ids[0] in [1/3, 2/3]
    return torch.cat((X[cof_ids, :], f_ids.unsqueeze(dim=-1)), dim=1)

# construct output vector for acquired points
def build_y_train(acquired_set):
    f_ids = get_f_ids(acquired_set)
    cof_ids = [a[1].to(dtype=int) for a in acquired_set]
    return torch.tensor([y[f_id][cof_id] for f_id, cof_id in zip(f_ids, cof_ids)]).unsqueeze(-1)


# #### retreiving costs inccurred for acquired set

# In[ ]:


# construct vector to track cost of acquired points
# entry i is cost of acquired COF i
def build_cost(acquired_set):
    f_ids = get_f_ids(acquired_set)
    cof_ids = [a[1].to(dtype=int) for a in acquired_set]
    return torch.tensor([cost[f_id][cof_id] for f_id, cof_id in zip(f_ids, cof_ids)]).unsqueeze(-1)

# construct vector to track cost of acquired points
# entry i is cost of acquired COF i within a given fidelity_id
def build_cost_fidelity(acquired_set, fidelity_id):
    assert fidelity_id in [0, 1]
    f_ids = get_f_ids(acquired_set)
    cof_ids = [a[1].to(dtype=int) for a in acquired_set]
    return torch.tensor([cost[f_id][cof_id] for f_id, cof_id in zip(f_ids, cof_ids) if f_id == fidelity_id]).unsqueeze(-1)


# ### train surrogate model and retreive its predictions

# In[ ]:


# return trained surrogate model
def train_surrogate_model(X_train, y_train):
    model = SingleTaskMultiFidelityGP(
        X_train, 
        y_train, 
        linear_truncated=False, # RBF for features and Downsampling for Fidelities
        outcome_transform=Standardize(m=1), # m is the output dimension
        data_fidelities=[X_train.shape[1]-1]
    )   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

# return mean, standard deviation of posterior acc to surrogate model
def mu_sigma(model, X, fidelity):
    assert fidelity in [1/3, 2/3]
    nb_COFs_here = X.size()[0]
    f = torch.tensor((), dtype=torch.float64).new_ones((nb_COFs_here, 1)) * fidelity
    X_f = torch.cat((X, f), dim=1) # last col is associated fidelity
    f_posterior = model.posterior(X_f)
    return f_posterior.mean.squeeze().detach().numpy(), np.sqrt(f_posterior.variance.squeeze().detach().numpy())


# #### Acquisition Function

# In[ ]:


# ratio of mean cost of sims of high fidelity to those of fidelity-fidelity, within acquired set so far.
def estimate_cost_ratio(acquired_set, fidelity):
    assert fidelity in [1/3, 2/3]
    f_id = get_f_ids(torch.tensor(fidelity))
    avg_cost_f  = torch.mean(build_cost_fidelity(acquired_set, f_id))
    avg_cost_hf = torch.mean(build_cost_fidelity(acquired_set, 1))
    cr = avg_cost_hf / avg_cost_f
    return cr.item()


# In[ ]:


# get the current best y-value of desired_fidelity in the acquired set
def get_y_max(acquired_set, fidelity):
    assert fidelity in [0, 1]
    f_ids = get_f_ids(acquired_set)
    cof_ids = [a[1].to(dtype=int) for a in acquired_set]
    return np.max([y[f_id][cof_id] for f_id, cof_id in zip(f_ids, cof_ids) if f_id == fidelity])


# In[ ]:


###
#  efficient multi-fidelity correlation function
#  corr(y at given fidelity, y at high-fidelity)
#  (see notes)
###
def mfbo_correlation_function(model, X, fidelity):
    assert fidelity in [1/3, 2/3]
    # given fidelity
    f   = torch.tensor((), dtype=torch.float64).new_ones((nb_COFs, 1)) * fidelity
    X_f = torch.cat((X, f), dim=1) # last col is associated fidelity
    
    #  high-fidelity
    hf   = torch.tensor((), dtype=torch.float64).new_ones((nb_COFs, 1)) * discrete_fidelities[-1]
    X_hf = torch.cat((X, hf), dim=1) # last col is associated fidelity

    # combine into a single tensor
    X_all_fid = torch.cat((X_f, X_hf), dim=0)
    
    # get variance for each fidelity
    var_f = torch.flatten(model.posterior(X_f).variance)
    var_hf = torch.flatten(model.posterior(X_hf).variance) # variance
    
    # posterior covariance 
    cov = torch.diag(model(X_all_fid).covariance_matrix[:X_f.size()[0], X_f.size()[0]:])
    
    corr = cov / (torch.sqrt(var_f) * torch.sqrt(var_hf))
    return corr


# In[ ]:


###CHANGE HERE ONLY 
#EI_hf is UCB now

def EI_hf(model, X, acquired_set):
    hf_mu, hf_sigma = mu_sigma(model, X, discrete_fidelities[-1])
    
    # Upper Confidence Bound (UCB)
    beta = 2  # exploration parameter - adjust as needed
    ucb = hf_mu + np.sqrt(beta) * hf_sigma
    return np.maximum(ucb, np.zeros(X.size()[0]))


###
#  acquisition function
###
def acquisition_scores(model, X, fidelity, acquired_set):
    assert fidelity in [1/3, 2/3]
    # expected improvement for high-fidelity
    ei = EI_hf(model, X, acquired_set) 
    
    # augmenting functions
    corr_f1_f0 = mfbo_correlation_function(model, X, fidelity)
    
    cr = estimate_cost_ratio(acquired_set, fidelity)

    scores = torch.from_numpy(ei) * corr_f1_f0 * cr
    return scores.detach().numpy()


# In[ ]:


# return True if (f_id, cof_id) in acquired set and False otherwise
def in_acquired_set(f_id, cof_id, acquired_set):
    assert f_id in [0, 1]
    fidelity = discrete_fidelities[f_id]
    for this_fidelity, this_cof_id in acquired_set:
        if this_cof_id == cof_id and this_fidelity == fidelity:
            return True
    return False


# #### tests

# In[ ]:


#most of these functions operate on an acquired_set which will look like this:
bogus_acquired_set = torch.tensor([[2/3, 10], [1/3, 3], [1/3, 4], [2/3, 50]])

###
#   in_acquired_set
###
assert not in_acquired_set(0, 10, bogus_acquired_set)
assert in_acquired_set(1, 10, bogus_acquired_set)
assert in_acquired_set(1, 50, bogus_acquired_set)
assert in_acquired_set(0, 3, bogus_acquired_set)
assert not in_acquired_set(0, 13, bogus_acquired_set)

###
#   build_X_train
###
bogus_X_train = build_X_train(bogus_acquired_set)

# first 14 are features.
np.allclose(X[[10, 3, 4, 50], :], bogus_X_train[:, :14])

# last one is fidelity param
assert np.allclose(bogus_X_train[:, 14], [2/3, 1/3, 1/3, 2/3])

###
#   build_y_train
###
bogus_y_train = build_y_train(bogus_acquired_set)
assert bogus_y_train[0] == y[1][10] # y[fid_id][cof_id]
assert bogus_y_train[1] == y[0][3]
assert bogus_y_train[2] == y[0][4]
assert bogus_y_train[3] == y[1][50]
                                   
###
#   get_f_ids
###
assert np.array_equal(get_f_ids(bogus_acquired_set), torch.tensor([1, 0, 0, 1]))

###
#  get_y_maxes_hf_acquired, get_y_max
###
assert np.all(get_y_maxes_hf_acquired(bogus_acquired_set) == np.array([y[1][10].item(), y[1][10].item(), y[1][10].item(), y[1][50].item()]))
assert get_y_max(bogus_acquired_set, 1) == y[1][50]
assert get_y_max(bogus_acquired_set[:2], 1) == y[1][10]
assert get_y_max(bogus_acquired_set[:], 0) == y[0][3]

###
#   accumulated_cost, build_cost, build_cost_fidelity
###
assert np.allclose(build_cost(bogus_acquired_set).squeeze(), np.array([cost[1][10], cost[0][3], cost[0][4], cost[1][50]]))
assert np.all(accumulated_cost(bogus_acquired_set) == np.array([cost[1][10], cost[1][10]+cost[0][3], cost[1][10]+cost[0][3]+cost[0][4], cost[1][10]+cost[0][3]+cost[0][4]+cost[1][50]]))
assert np.allclose(build_cost_fidelity(bogus_acquired_set, 1).squeeze(), np.array([cost[1][10], cost[1][50]]))
assert np.allclose(build_cost_fidelity(bogus_acquired_set, 0).squeeze(), np.array([cost[0][3], cost[0][4]]))


###
#   estimate_cost_ratio
###
assert estimate_cost_ratio(bogus_acquired_set, 2/3) == 1 # high to high
assert estimate_cost_ratio(bogus_acquired_set, 1/3) > 1# high to low
assert estimate_cost_ratio(bogus_acquired_set, 1/3) == (cost[1][10] + cost[1][50]) / (cost[0][3] + cost[0][4])

###
#   calc_fidelity_fraction
###
assert np.allclose(calc_fidelity_fraction(bogus_acquired_set, 1/3), np.array([0.0, 1/2, 2/3, 2/4]))
assert np.allclose(calc_fidelity_fraction(bogus_acquired_set, 2/3), np.array([1.0, 1/2, 1/3, 2/4]))

###
#   EI_hf
###
bogus_X_train_hf = torch.clone(bogus_X_train) # all need to be hf
bogus_X_train_hf[1][-1] = 2/3
bogus_X_train_hf[2][-1] = 2/3

# train bogus model
bogus_model = train_surrogate_model(bogus_X_train, bogus_y_train) 

# use BO Torch's acquisition functions
bot_ucb = UpperConfidenceBound(bogus_model, beta=2)
bot_ucb_vals = bot_ucb.forward(bogus_X_train_hf.unsqueeze(1))

# ours
our_ucb = EI_hf(bogus_model, X[[10, 3, 4, 50], :], bogus_acquired_set)
assert np.allclose(bot_ucb_vals.detach().numpy(), our_ucb)


# In[ ]:


# # Debug: print the values to see what's different, written by AI
# print("BoTorch UCB values:", bot_ei_vals.detach().numpy())
# print("Our UCB values:", our_ei)
# print("Difference:", bot_ei_vals.detach().numpy() - our_ei)
# print("Shape of BoTorch output:", bot_ei_vals.shape)
# print("Shape of our output:", our_ei.shape if hasattr(our_ei, 'shape') else type(our_ei))


# ### Bayesian Algorithm

# In[ ]:


def run_Bayesian_optimization(nb_iterations, initializing_COFs, verbose=False, stop_after_top_acquired=True):
    assert nb_iterations > len(initializing_COFs)
    ###
    #  initialize acquired set
    ###
    acquired_set = initialize_acquired_set(initializing_COFs, discrete_fidelities)
    
    ###
    #  analyze-plan-simulate iterations
    ###
    for i in range(nb_COFs_initialization * len(discrete_fidelities), nb_iterations): 
        print("BO iteration: ", i)
        ###
        #  construct training data (perform experiments)
        ###
        X_train = build_X_train(acquired_set)
        y_train = build_y_train(acquired_set)

        if verbose:
            print("Initialization - \n")
            print("\tCOF IDs acquired    = ", [acq_[1].item() for acq_ in acquired_set])
            print("\tfidelities acquired = ", [acq_[0].item() for acq_ in acquired_set])
            print("\tcosts acquired      = ", build_cost(acquired_set), " [min]")

            print("\n\tTraining data:\n")
            print("\t\t X train shape = ", X_train.shape)
            print("\t\t y train shape = ", y_train.shape)
            print("\t\t training feature vector = \n", X_train)
        
        ###
        #  train Model
        ###
        model = train_surrogate_model(X_train, y_train)
        
        ###
        #  acquire new (COF, fidelity) not yet acquired.
        ###
        # entry (fid_id, cof_id) is the acquisition value for fidelity f_id and cof cof_id
        the_acquisition_scores = np.array([acquisition_scores(model, X, fidelity, acquired_set) for fidelity in discrete_fidelities])
        
        # overwrite acquired COFs/fidelities with negative infinity to not choose these.
        for fidelity, cof_id in acquired_set:
            the_acquisition_scores[get_f_ids(fidelity), cof_id.to(dtype=int)] = - np.inf
        
        # select COF/fidelity with highest aquisition score.
        f_id, cof_id = np.unravel_index(np.argmax(the_acquisition_scores), np.shape(the_acquisition_scores))
        assert f_id in [0, 1]
        assert not in_acquired_set(f_id, cof_id, acquired_set)
        assert np.max(the_acquisition_scores) == the_acquisition_scores[f_id, cof_id]
        
        # update acquired_set
        acq = torch.tensor([[discrete_fidelities[f_id], cof_id]]) # dtype=int
        acquired_set = torch.cat((acquired_set, acq))

        ###
        #  print useful info
        ###
        if verbose:
            print("\tacquired COF ", cof_id, " at fidelity, ", f_id)
            print("\t\ty = ", y[f_id][cof_id].item())
            print("\t\tcost = ", cost[f_id][cof_id])
            
        if stop_after_top_acquired:
            cof_id_with_max_selectivity = np.argmax(y[1])
            if cof_id_with_max_selectivity == cof_id and f_id == 1:
                print("found top COF! exiting.")
                return acquired_set
        
    return acquired_set


# # Run MFBO

# In[ ]:


###
#  construct initial inputs
###
nb_COFs_initialization = 3   # at each fidelity, number of COFs to initialize with
nb_iterations = 2 * nb_COFs  # BO budget, includes initializing COFs. this is actually max # iterations

# if ablation_study_flag:
#     print("ablation study: {}".format(ablation_study_flag))
#     # the maximum possible number itterations = num_fidelities * nb_COFs
#     # this would efectively constitute a low-fidelity exhaustive search 
#     # followed by a high-fidelity exhaustive search
#     nb_iterations = 2 * nb_COFs 
#     print("max. number of iterations: {}".format(nb_iterations))


# run once, using `init_cof_ids[0]`, which is the COF closest to the mean of the features.

# In[ ]:


if not ablation_study_flag:
    acquired_set = run_Bayesian_optimization(nb_iterations, init_cof_ids[0], verbose=False)


# In[ ]:


# unpack search results
f_ids   = [acquired_set[i][0].item()      for i in range(len(acquired_set))]
cof_ids = [int(acquired_set[i][1].item()) for i in range(len(acquired_set))]

# which COF has the largest high-fidelity selectivity?
cof_id_with_max_hi_fid_selectivity = np.argmax(y[1]).item()

# iteration we found top COF
n_iter_top_cof_found = np.where([cof_ids[i] == cof_id_with_max_hi_fid_selectivity and f_ids[i] > 0.5 for i in range(len(cof_ids))])[0].item()
n_iter_top_cof_found 

#this is the iteration no. in BO run where the top COF was observed at high fidelity


# In[ ]:


#COF index of the top-performing COF 

cof_id_with_max_hi_fid_selectivity


# ### observe status of surrogate model's knowledge the iteration before the top COF was acquired.

# In[ ]:


# find COFs that are simualted in with high- and low-fidelity.
hi_fid_cofs = [cof_ids[i] for i in range(n_iter_top_cof_found) if f_ids[i] > 0.5]
lo_fid_cofs = [cof_ids[i] for i in range(n_iter_top_cof_found) if f_ids[i] < 0.5]
# find COFs simulated at both fidelities
ids_cofs_hi_and_lo_fid = np.intersect1d(hi_fid_cofs, lo_fid_cofs)
ids_cofs_hi_and_lo_fid


# the correlation between high- and low-fidelity selectivities. only pertains to those COFs with both simulated hi and lo.

# In[ ]:


# build selectivity array for plotting, 
y_los = [y[0][c].item() for c in ids_cofs_hi_and_lo_fid]
y_his = [y[1][c].item() for c in ids_cofs_hi_and_lo_fid]

fig = plt.figure()
plt.plot([0, 25], [0, 25], linestyle="--", color="k", linewidth=1)
plt.scatter(y_los, y_his, zorder=10)
ax = plt.gca()
plt.xlim(0, 25)
plt.ylim(0, 25)
ax.set_aspect("equal", "box")
plt.xlabel("low-fidelity Xe/Kr selectivity")
plt.ylabel("high-fidelity Xe/Kr selectivity")
plt.tight_layout()
if not ablation_study_flag:
    plt.savefig("../figs/lo_vs_hi_fid_selectivity.pdf", format='pdf')
plt.show()


# In[ ]:


# get COF ids not in acquired set with high-fidelity sims. these are test COFs for high-fidelity standpoint.
test_cof_ids = [cof_id for cof_id in range(nb_COFs) if not (cof_id in hi_fid_cofs)]
len(test_cof_ids)


# In[ ]:


cof_id_with_max_hi_fid_selectivity in test_cof_ids # the COF with the highest selectivity should be in here. cuz we didn't acquire it yet.


# In[ ]:


id_in_test_cofs_of_opt_cof = np.where([c == cof_id_with_max_hi_fid_selectivity for c in test_cof_ids])[0].item()
id_in_test_cofs_of_opt_cof


# In[ ]:


# train surrogate model for test data, on acquired set up till top COF was found.
X_train = build_X_train(acquired_set[:n_iter_top_cof_found])
y_train = build_y_train(acquired_set[:n_iter_top_cof_found])

X_test = X[test_cof_ids, :]

model = train_surrogate_model(X_train, y_train)

# get model predictions on test COFs, for high-fidelity.
y_pred, sigma = mu_sigma(model, X_test, discrete_fidelities[-1])

# plot true vs predicted
y_true = [y[1][c].item() for c in test_cof_ids]

r2 = r2_score(y_true, y_pred)
abserr = mean_absolute_error(y_true, y_pred)

###
#  parity plot
###
gridspec_kw={'width_ratios': [6, 2], 'height_ratios': [2, 6]} # set ratios
fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=gridspec_kw, figsize=(8, 8))
# fig = plt.figure()
ax[0, 1].axis("off")
ax[1, 0].plot([0, 20], [0, 20], linestyle="--", color="k", linewidth=1)
# ax = plt.gca()
ax[1, 0].set_xlim(0, 20)
ax[1, 0].set_ylim(0, 20)

#ax[1, 0].set_aspect("equal", "box")

ax[1, 0].text(5, 15, "R$^2$=%.2f\nMAE=%.2f" % (r2, abserr))
ax[1, 0].scatter(y_true, y_pred, fc='none', ec="k")
ax[1, 0].set_xlabel("true\nhigh-fidelity Xe/Kr selectivity")
ax[1, 0].set_ylabel("predicted\nhigh-fidelity Xe/Kr selectivity")
# plot acquired COF
ax[1, 0].scatter(y_true[id_in_test_cofs_of_opt_cof], y_pred[id_in_test_cofs_of_opt_cof], marker="x", color="red")


###
#  histogram of selectivities
###
hist_color = sns.color_palette("husl", 8)[7]
ax[0, 0].hist(y_true, color=hist_color, alpha=0.5) # 
ax[0, 0].sharex(ax[1, 0])
ax[0, 0].set_ylabel('# COFs')
plt.setp(ax[0, 0].get_xticklabels(), visible=False) # remove yticklabels

hist_color = sns.color_palette("husl", 8)[7]
ax[1, 1].hist(y_pred, color=hist_color, alpha=0.5, orientation="horizontal") # 
ax[1, 1].sharey(ax[1, 0])
ax[1, 1].set_xlabel('# COFs')
plt.setp(ax[1, 1].get_yticklabels(), visible=False) # remove yticklabels

sns.despine()
plt.tight_layout()
if not ablation_study_flag:
    plt.savefig("../figs/surrogate_parity_with_hist.pdf", format="pdf")

plt.show()


# for kicks, compare to RF.

# In[ ]:


# rf = RandomForestRegressor()
# rf.fit(X_train[:, :14], y_train)
# rf.score(X_test, y_true)


# In[ ]:


# ids_sorted = np.argsort(y_true)[::-1]

# plt.figure(figsize=(12, 3))
# plt.errorbar(range(len(y_true)), y_pred[ids_sorted], yerr=sigma, linewidth=1, marker="o")
# plt.errorbar(0, y_pred[id_in_test_cofs_of_opt_cof], yerr=sigma[id_in_test_cofs_of_opt_cof], linewidth=1, marker="o", color="red")
# plt.xlabel("rank according to true high-fidelity Xe/Kr selectivity")
# plt.ylabel("predicted\nhigh-fidelity\nXe/Kr selectivity")
# plt.xlim(-1, len(test_cof_ids) +1)
# plt.tight_layout()
# if not ablation_study_flag:
#     plt.savefig("../figs/surrogate_predictions.pdf", format="pdf")

# plt.show()


# In[ ]:


# np.where([cof_id == cof_id_with_max_hi_fid_selectivity for cof_id in test_cof_ids])


# # Run MFBO under different initializations

# In[ ]:


def save_run_results(acquired_set, run, ablation_study_flag):
    # compute attributes of acquired set
    y_acquired    = build_y_train(acquired_set)
    y_maxes_acq   = get_y_maxes_hf_acquired(acquired_set.detach().numpy())
    fid_fraction  = calc_fidelity_fraction(acquired_set.detach().numpy(), discrete_fidelities[1])
    cost_acquired = build_cost(acquired_set)
    acc_cost      = accumulated_cost(acquired_set)
    
    # when did MFBO recover top COF?
    cof_id_with_max_selectivity = np.argmax(y[1])
    BO_iter_top_cof_acquired = float("inf") # dummy 
    for i, (f_id, cof_id) in enumerate(acquired_set):
        if cof_id.to(dtype=int) == cof_id_with_max_selectivity and get_f_ids(f_id) == 1:
            BO_iter_top_cof_acquired = i
            break
        elif i == len(acquired_set) - 1:
            print("oh no, top COF not acquired!")
    
    mfbo_res = dict({
        'acquired_set': acquired_set.detach().numpy(),
         'y_acquired': y_acquired.detach().numpy(),
         'y_max_acquired': y_maxes_acq,
         'fidelity_fraction': fid_fraction,
         'cost_acquired': cost_acquired.flatten().detach().numpy(),
         'accumulated_cost': acc_cost / 60,
         'nb_COFs_initialization': nb_COFs_initialization,
         'BO_iter_top_cof_acquired': BO_iter_top_cof_acquired,
         'elapsed_time (min)': 0.0,
         'post_preprint': True
        })
    
    import os  
    import pickle
    
    # Dynamically compute path from script location (works on any system)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # go up from run_BO to project root
    output_folder = os.path.join(project_root, 'search_results', 'mfbo', 'Run5_10Dec_UCB_B2')
    os.makedirs(output_folder, exist_ok=True)

    pickle_filename = os.path.join(output_folder, 'mfbo_results_run_{}'.format(run))
    if ablation_study_flag:
        pickle_filename += "_ablation"
    pickle_filename += ".pkl"
    with open(pickle_filename, 'wb') as file:
        pickle.dump(mfbo_res, file)


# In[ ]:


###
#  run search
###
for j in range(nb_runs):
    initializing_COFs = init_cof_ids[j]
    if ablation_study_flag:
        # each time, randomly shuffle features within a column
        for k in range(X.size()[1]):
            row_ids = torch.randperm(X.size()[0])
            X[:, k] = X[row_ids, k]

    # check the length of each initializing set
    assert len(initializing_COFs) == nb_COFs_initialization
    print("run #: {}".format(j))

    ###
    #  run BO search
    ###
    acquired_set = run_Bayesian_optimization(nb_iterations, initializing_COFs)
    save_run_results(acquired_set, j, ablation_study_flag)


# In[ ]:


# nb_runs

