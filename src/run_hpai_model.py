import hpai_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
seed = 0
np.random.seed(seed)

# Choose options for running the model
run_mcmc = False # Run an MCMC chain
use_mcmc = True # Use posterior distributions from MCMC chains
run_intervention = True # Include interventions in model projections
run_projections = True  # Run model projections
use_projections = True # Load previously saved model projections
sellke = True # Use Sellke construction for model projections otherwise stochastic tau-leaping simulations
plots = True # Generate plots

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

# Get data structure for the model
data = hpai_model.Data(data_input)
print("Data loaded")

# Set model parameters
model = hpai_model.ModelStructure(data)

# Set up model fitting
modelfit = hpai_model.ModelFitting(model, total_iterations=211000, burn_in=11000, single_iterations=1000)

# Run the model for one chain
if run_mcmc:
    modelfit.run_mcmc_chain(chain_number=1, save_iter=np.array([2000, 11000, 51000, 211000]))

# Load MCMC chains
if use_mcmc:
    modelfit.load_chains(chain_numbers=[1,2])

# Plot MCMC results
if run_mcmc or use_mcmc:
    modelplotting = hpai_model.Plotting(modelfit)
    if plots:
        modelplotting.plot_parameter_chains()
        modelplotting.plot_parameter_posteriors()

# Set up model projections
if run_mcmc or use_mcmc:
    modelsim = hpai_model.ModelSimulator(model_fitting=modelfit, reps=10, sellke=sellke)
else:
    modelsim = hpai_model.ModelSimulator(model_structure=model, reps=10, sellke=sellke)

# Run or load model projections
if run_projections or use_projections:
    if run_projections:
        if run_intervention:
            vaccine = hpai_model.Intervention()
        else:
            vaccine = None
        modelsim.run_model(intervention=vaccine)
    else:
        if run_intervention:
            vaccine = hpai_model.Intervention()
        else:
            vaccine = None
        modelsim.load_projections(intervention=vaccine)

    # Plot model projections
    if plots:
        modelplotting_sim = hpai_model.Plotting(model_simulator=modelsim)
        modelplotting_sim.plot_projections()



