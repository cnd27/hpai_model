import model_nice as hpai_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

seed = 0
np.random.seed(seed)
run_mcmc = False
data_start = pd.to_datetime('2022-10-01')
data_end = pd.to_datetime('2023-09-30')

# List locations of data files
data_paths = hpai_model.DataFilePaths(premises='../HPAI_2025/Data/HPAI_Farm_Data_2_update2.txt',
                                      cases='../HPAI_2025/Data/case_data_20230424_update2.xlsx',
                                      match='../HPAI_2025/Data/MatchedFarms_update2.txt',
                                      regions='../HPAI_2025/Data/NUTS1_Jan_2018_SGCB_in_the_UK.shp',
                                      counties='../HPAI_2025/Data/CTYUA_MAY_2023_UK_BGC.shp')

# Format data files into required format
data_input = hpai_model.DataLoader(data_paths, data_start, data_end)

# Get data structure for model
data = hpai_model.Data(data_input)
print("Data loaded")

# Set model parameters
model = hpai_model.ModelStructure(data)
# modelfit = hpai_model.ModelFitting(model, total_iterations=12000, burn_in=1000)
# # Run the model
# if run_mcmc:
#     modelfit.run_mcmc_chain(save_iter=np.array([2, 11000, 51000, 210000]))
# else:
#     modelfit.load_chains(chain_numbers=[0])
modelsim = hpai_model.ModelSimulator(model, reps=2)
modelsim.run_model()
modelsim.save_projections()

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