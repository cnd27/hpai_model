import hpai_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

seed = 0
np.random.seed(seed)
run_mcmc = False
use_mcmc = False
run_projections = False
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
modelfit = hpai_model.ModelFitting(model)

# Run the model
if run_mcmc:
    modelfit.run_mcmc_chain(save_iter=np.array([11000, 51000, 210000]))

if use_mcmc:
    modelfit.load_chains(chain_numbers=[0])
    modelsim = hpai_model.ModelSimulator(modelfit, reps=2)
else:
    modelsim = hpai_model.ModelSimulator(model, reps=2)

if run_projections:
    modelsim.run_model()
    modelsim.save_projections()
else:
    modelsim.load_projections()

modelplotting = hpai_model.Plotting(modelfit, modelsim)
modelplotting.plot_projections()
plt.show()

# fig, ax = plt.subplots(4, 4, figsize=(16, 16))
# for i in range(16):
#     row = i // 4
#     col = i % 4
#     ax[row, col].hist(modelfit.parameter_posterior[:, i])
# plt.tight_layout()
# fig, ax = plt.subplots(4, 4, figsize=(16, 16))
# for i in range(16):
#     row = i // 4
#     col = i % 4
#     ax[row, col].plot(modelfit.parameter_chains[0, i, :])
# plt.tight_layout()
# plt.show()


print(1)