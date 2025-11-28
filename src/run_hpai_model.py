import hpai_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
seed = 0
np.random.seed(seed)

# Choose options for running the model
run_mcmc = False
use_mcmc = True
run_projections = True
sellke = True
plots = True

# Define time period
data_start = pd.to_datetime('2022-10-01')
data_end = pd.to_datetime('2023-09-30')

# List locations of data files
data_paths = hpai_model.DataFilePaths(premises='../data/HPAI_Farm_Data_2_update2.txt',
                                      cases='../data/case_data_20230424_update2.xlsx',
                                      match='../data/MatchedFarms_update2.txt',
                                      regions='../data/NUTS1_Jan_2018_SGCB_in_the_UK.shp',
                                      counties='../data/CTYUA_MAY_2023_UK_BGC.shp')

# Format data files into required format
data_input = hpai_model.DataLoader(data_paths, data_start, data_end)

# Get data structure for model
data = hpai_model.Data(data_input)
print("Data loaded")

# Set model parameters
model = hpai_model.ModelStructure(data)
modelfit = hpai_model.ModelFitting(model, total_iterations=211000, burn_in=11000, single_iterations=1000)

# Run the model
if run_mcmc:
    modelfit.run_mcmc_chain(save_iter=np.array([11000, 51000, 210000]))

# Load MCMC chains
if use_mcmc:
    modelfit.load_chains(chain_numbers=[1,2])

# Plot MCMC results
if run_mcmc or use_mcmc:
    modelplotting = hpai_model.Plotting(modelfit)
    if plots:
        modelplotting.plot_parameter_chains()
        modelplotting.plot_parameter_posteriors()

# Get model projections
if run_mcmc or use_mcmc:
    modelsim = hpai_model.ModelSimulator(model_fitting=modelfit, reps=10, sellke=sellke)
else:
    modelsim = hpai_model.ModelSimulator(model_structure=model, reps=10, sellke=sellke)
if run_projections:
    modelsim.run_model()
else:
    modelsim.load_projections()

# Plot model projections
if plots:
    modelplotting_sim = hpai_model.Plotting(model_simulator=modelsim)
    modelplotting_sim.plot_projections()

