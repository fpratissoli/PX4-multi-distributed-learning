import numpy as np
import matplotlib.pyplot as plt
import utilities as utils
from pathlib import Path
from scipy.interpolate import griddata
import xarray as xr


# np.random.seed(0)

# area_size = 40
# x_inf, y_inf = 0, 0
# x_sup, y_sup = area_size, area_size
# BBOX = [x_inf, y_inf, x_sup, y_sup]
# d_field_ = 1
# x1_ = np.arange(0, area_size + d_field_, d_field_)
# x2_ = np.arange(0, area_size + d_field_, d_field_)
# _X1, _X2 = np.meshgrid(x1_, x2_)
# mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# robNum = 6

# sensRange = np.sqrt(((x_sup * y_sup) * 1.0 / robNum) / np.pi) * 2

# # Generate random means
# peaks = 2 # np.random.randint(1, 10)
# means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
# sigma = 6
# Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
# Z = Z.reshape(len(x1_), len(x2_))


""" Dataset """
# Generate data
np.random.seed(0)

area_size = 40
x_inf, y_inf = 0, 0
x_sup, y_sup = area_size, area_size
BBOX = [x_inf, y_inf, x_sup, y_sup]
d_field_ = 1
x1_ = np.arange(0, area_size, d_field_)
x2_ = np.arange(0, area_size, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Load the dataset and select the region of interest
dataset_path = Path('dataset.nc')
ds = xr.open_dataset(dataset_path)
sst_region = ds['analysed_sst'].sel(lat=slice(22, 23), lon=slice(-115, -114)).isel(time=0).values

# Convert from Kelvin to Celsius
sst_region = (sst_region - 273.15).astype(np.float32)

# Define original and target grids
original_shape = sst_region.shape
target_size = (area_size, area_size)
original_grid = np.meshgrid(np.arange(original_shape[1]), np.arange(original_shape[0]))
target_grid = np.meshgrid(np.linspace(0, original_shape[1]-1, target_size[1]),
                          np.linspace(0, original_shape[0]-1, target_size[0]))

# Interpolate SST data to the target grid
sst_interpolated = griddata(
    np.column_stack([g.flatten() for g in original_grid]),
    sst_region.flatten(),
    np.column_stack([g.flatten() for g in target_grid]),
    method='linear'
).reshape(target_size)

# Min-Max Normalization
field = (sst_interpolated - sst_interpolated.min()) / (sst_interpolated.max() - sst_interpolated.min())

robNum = 20

sensRange = np.sqrt(((x_sup * y_sup) * 1.0 / robNum) / np.pi) * 2
print(sensRange)

# Load data
path_gen = Path('proposed-sims')
sims = ['sim-1', 'sim-2', 'sim-3']
rmseHistory_all = []
nlpdHistory_all = []
observedHistory_all = []
filteredHistory_all = []
DEC_gapx_time_hist_all = []
DAC_time_hist_all = []
timeHistory_all = []
meanHistory_all = []
stdHistory_all = []
robotHistory_all = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_all.append(rmseHistory)
    robotHistory_all.append(robotHistory)
    nlpdHistory_all.append(nlpdHistory)
    observedHistory_all.append(observedHistory)
    filteredHistory_all.append(filteredHistory)
    DEC_gapx_time_hist_all.append(DEC_gapx_time_hist)
    DAC_time_hist_all.append(DAC_time_hist)
    timeHistory_all.append(timeHistory)
    meanHistory_all.append(meanHistory)
    stdHistory_all.append(stdHistory)

# RAL Data

# path_gen = Path('ral-sims')
# sims_ral = ['sim-1', 'sim-2', 'sim-3']
# rmseHistory_ral = []
# nlpdHistory_ral = []
# observedHistory_ral = []
# filteredHistory_ral = []
# DEC_gapx_time_hist_ral = []
# DAC_time_hist_ral = []
# timeHistory_ral = []
# meanHistory_ral = []
# stdHistory_ral = []

# for sim in sims_ral:
#     path = path_gen / sim
#     robotHistory = np.load(path / "robotHistory.npy")
#     rmseHistory = np.load(path / "rmseHistory.npy")
#     nlpdHistory = np.load(path / "nlpdHistory.npy")
#     observedHistory = np.load(path / "observedHistory.npy")
#     filteredHistory = np.load(path / "filteredHistory.npy")
#     DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
#     DAC_time_hist = np.load(path / "DAC_time_hist.npy")
#     timeHistory = np.load(path / "timeHistory.npy")
#     meanHistory = np.load(path / "meanHistory.npy")
#     stdHistory = np.load(path / "stdHistory.npy")

#     rmseHistory_ral.append(rmseHistory)
#     nlpdHistory_ral.append(nlpdHistory)
#     observedHistory_ral.append(observedHistory)
#     filteredHistory_ral.append(filteredHistory)
#     DEC_gapx_time_hist_ral.append(DEC_gapx_time_hist)
#     DAC_time_hist_ral.append(DAC_time_hist)
#     timeHistory_ral.append(timeHistory)
#     meanHistory_ral.append(meanHistory)
#     stdHistory_ral.append(stdHistory)


# No Filter Data
path_gen = Path('proposed-no-nystrom')
sims_nofilter = ['sim-1', 'sim-2', 'sim-3']
timeHistory_nofilter = []
rmseHistory_nofilter = []
nlpdHistory_nofilter = []

for sim in sims_nofilter:
    path = path_gen / sim
    timeHistory = np.load(path / "timeHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")

    timeHistory_nofilter.append(timeHistory)
    nlpdHistory_nofilter.append(nlpdHistory)
    rmseHistory_nofilter.append(rmseHistory)


robotHistory = robotHistory_all[0]

fig = plt.figure(figsize=(10, 5))
# Inital position of the robots
ax = fig.add_subplot(121, aspect='equal')
ax.contourf(np.linspace(0, target_size[1], target_size[1]),
                np.linspace(0, target_size[0], target_size[0]),
                field, cmap='YlOrRd')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_xlim([0, target_size[1]])
ax.set_ylim([0, target_size[0]])
ax.set_title('Robot paths')
fontdict = {'weight': 'bold', 'size': 12, 'color': 'black'}

for i in range(robNum):
    ax.scatter(robotHistory[i, 0, 0], robotHistory[i, 1, 0], marker='o', label=f'Robot {i}')
    ax.text(robotHistory[i, 0, 0] + 1, robotHistory[i, 1, 0] + 1, f'{i}', ha='center', va='center', fontdict=fontdict)

# Plot the voronoi regions
limRegions = utils.voronoi_alg_limited(robotHistory[:, :, 0], BBOX, sensRange)
for i, region in enumerate(limRegions):
    # Extract the exterior coordinates of the Polygon
    x, y = region.exterior.xy
    ax.plot(x, y, color="black", linewidth=2)
ax.grid(True, alpha=0.3)


ax = fig.add_subplot(122, aspect='equal')
ax.contourf(np.linspace(0, target_size[1], target_size[1]),
             np.linspace(0, target_size[0], target_size[0]),
             field, cmap='YlOrRd')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_xlim([0, target_size[1]])
ax.set_ylim([0, target_size[0]])
ax.set_title('Robot paths')
for i in range(robNum):
    ax.plot(robotHistory[i, 0, :], robotHistory[i, 1, :])
    ax.scatter(robotHistory[i, 0, 0], robotHistory[i, 1, 0], s=80, facecolors='none', edgecolors=f'C{i}')
    ax.scatter(robotHistory[i, 0, -1], robotHistory[i, 1, -1], marker='o', label=f'Robot {i}')
    ax.text(robotHistory[i, 0, -1] + 1, robotHistory[i, 1, -1] + 1, f'{i}', ha='center', va='center', fontdict=fontdict)

# Plot the voronoi regions
limRegions = utils.voronoi_alg_limited(robotHistory[:, :, -1], BBOX, sensRange)
for i, region in enumerate(limRegions):
    # Extract the exterior coordinates of the Polygon
    x, y = region.exterior.xy
    ax.plot(x, y, color="black", linewidth=2)
ax.grid(True, alpha=0.3)
plt.show()

# # Plot the RMSE mean    
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
rmse_mean = np.mean(rmseHistory_all, axis=0)
rmse_std = np.std(rmseHistory_all, axis=0)
ax.plot(np.mean(rmse_mean, axis=0), label='Mean RMSE')
ax.fill_between(np.arange(rmse_mean.shape[1]), rmse_mean[0] - rmse_std[0], rmse_mean[0] + rmse_std[0], alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('RMSE')
ax.set_title('RMSE')
ax.legend()
ax.grid(alpha=0.3)

# # Plot the NLPD mean
ax = fig.add_subplot(122)
nlpd_mean = np.mean(nlpdHistory_all, axis=0)
nlpd_std = np.std(nlpdHistory_all, axis=0)
ax.plot(np.mean(nlpd_mean, axis=0), label='Mean NLPD')
ax.fill_between(np.arange(nlpd_mean.shape[1]), nlpd_mean[0] - nlpd_std[0], nlpd_mean[0] + nlpd_std[0], alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('NLPD')
ax.set_title('NLPD')
ax.legend()
ax.grid(alpha=0.3)

# # Plot the observed and filtered data
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
observe_mean = np.mean(np.mean(observedHistory_all, axis=0), axis=0)
observe_std = np.std(np.mean(observedHistory_all, axis=0), axis=0)
ax.plot(observe_mean, label='Mean RAL Filtered Samples', alpha=0.5)
ax.fill_between(np.arange(observe_mean.shape[0]), observe_mean - observe_std, observe_mean + observe_std, alpha=0.3)
filtered_mean = np.mean(np.mean(filteredHistory_all, axis=0), axis=0)
filtered_std = np.std(np.mean(filteredHistory_all, axis=0), axis=0)
ax.plot(filtered_mean, label='Mean Samples used in FL', linestyle='--')
ax.fill_between(np.arange(filtered_mean.shape[0]), filtered_mean - filtered_std, filtered_mean + filtered_std, alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('Data')
ax.set_title('Observed (RAL) and filtered data (used in FL)')
ax.legend()
ax.grid(alpha=0.3)

# # Plot the DEC gapx time history
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
DEC_gapx_time_hist_mean = np.mean(DEC_gapx_time_hist_all, axis=0).squeeze()
DEC_gapx_time_hist_std = np.std(DEC_gapx_time_hist_all, axis=0).squeeze()
ax.plot(DEC_gapx_time_hist_mean, label='Mean DEC gapx')
ax.fill_between(np.arange(DEC_gapx_time_hist_mean.shape[0]), DEC_gapx_time_hist_mean - DEC_gapx_time_hist_std, DEC_gapx_time_hist_mean + DEC_gapx_time_hist_std, alpha=0.3)
DAC_time_hist_mean = np.mean(DAC_time_hist_all, axis=0).squeeze()
DAC_time_hist_std = np.std(DAC_time_hist_all, axis=0).squeeze()
ax.plot(DAC_time_hist_mean, label='Mean DAC')
ax.fill_between(np.arange(DAC_time_hist_mean.shape[0]), DAC_time_hist_mean - DAC_time_hist_std, DAC_time_hist_mean + DAC_time_hist_std, alpha=0.3)
timeHistory_mean = np.mean(timeHistory_all, axis=0).squeeze()
timeHistory_std = np.std(timeHistory_all, axis=0).squeeze()
ax.plot(timeHistory_mean, label='Mean Time')
ax.fill_between(np.arange(timeHistory_mean.shape[0]), timeHistory_mean - timeHistory_std, timeHistory_mean + timeHistory_std, alpha=0.3)
ax.set_xlabel('Period')
ax.set_ylabel('Time')
ax.set_title('Time history')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# Plot the time history
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
time_mean = np.mean(timeHistory_all, axis=0).squeeze()
time_std = np.std(timeHistory_all, axis=0).squeeze()
ax.plot(time_mean, label='Mean Time')
ax.fill_between(np.arange(time_mean.shape[0]), time_mean - time_std, time_mean + time_std, alpha=0.3)
time_mean_nofilter = np.mean(timeHistory_nofilter, axis=0).squeeze()
time_std_nofilter = np.std(timeHistory_nofilter, axis=0).squeeze()
ax.plot(time_mean_nofilter, label='Mean Time (No Filter)', linestyle='--')
ax.fill_between(np.arange(time_mean_nofilter.shape[0]), time_mean_nofilter - time_std_nofilter, time_mean_nofilter + time_std_nofilter, alpha=0.3)
# ax.plot(timeHistory_mean_nofilter, label='Mean Time (No Filter)', linestyle='--')
# ax.fill_between(np.arange(timeHistory_mean_nofilter.shape[1]), timeHistory_mean_nofilter[0] - timeHistory_std_nofilter[0], timeHistory_mean_nofilter[0] + timeHistory_std_nofilter[0], alpha=0.3)
ax.set_xlabel('Period')
ax.set_ylabel('Comp. Time')
ax.set_title('Time history')
ax.legend()
ax.grid(alpha=0.3)

# Plot the RMSE mean
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
rmse_mean = np.mean(rmseHistory_all, axis=0)
rmse_std = np.std(rmseHistory_all, axis=0)
ax.plot(np.mean(rmse_mean, axis=0), label='Mean RMSE')
ax.fill_between(np.arange(rmse_mean.shape[1]), rmse_mean[0] - rmse_std[0], rmse_mean[0] + rmse_std[0], alpha=0.3)
rmse_mean_nofilter = np.mean(rmseHistory_nofilter, axis=0)
rmse_std_nofilter = np.std(rmseHistory_nofilter, axis=0)
ax.plot(np.mean(rmse_mean_nofilter, axis=0), label='Mean RMSE (No Filter)', linestyle='--')
ax.fill_between(np.arange(rmse_mean_nofilter.shape[1]), rmse_mean_nofilter[0] - rmse_std_nofilter[0], rmse_mean_nofilter[0] + rmse_std_nofilter[0], alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('RMSE')
ax.set_title('RMSE')
ax.legend()
ax.grid(alpha=0.3)

# Plot the NLPD mean
ax = fig.add_subplot(122)
nlpd_mean = np.mean(nlpdHistory_all, axis=0)
nlpd_std = np.std(nlpdHistory_all, axis=0)
ax.plot(np.mean(nlpd_mean, axis=0), label='Mean NLPD')
ax.fill_between(np.arange(nlpd_mean.shape[1]), nlpd_mean[0] - nlpd_std[0], nlpd_mean[0] + nlpd_std[0], alpha=0.3)
nlpd_mean_nofilter = np.mean(nlpdHistory_nofilter, axis=0)
nlpd_std_nofilter = np.std(nlpdHistory_nofilter, axis=0)
ax.plot(np.mean(nlpd_mean_nofilter, axis=0), label='Mean NLPD (No Filter)', linestyle='--')
ax.fill_between(np.arange(nlpd_mean_nofilter.shape[1]), nlpd_mean_nofilter[0] - nlpd_std_nofilter[0], nlpd_mean_nofilter[0] + nlpd_std_nofilter[0], alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('NLPD')
ax.set_title('NLPD')
ax.legend()
ax.grid(alpha=0.3)
plt.show()


# Time comparison with RAL
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
time_mean = np.mean(timeHistory_all, axis=0).squeeze()
time_std = np.std(timeHistory_all, axis=0).squeeze()
ax.plot(time_mean, label='Mean Time Proposed')
ax.fill_between(np.arange(time_mean.shape[0]), time_mean - time_std, time_mean + time_std, alpha=0.3)
time_mean_ral = np.mean(timeHistory_ral, axis=0).squeeze()
time_std_ral = np.std(timeHistory_ral, axis=0).squeeze()
ax.plot(time_mean_ral, label='Mean Time RAL', linestyle='--')
ax.fill_between(np.arange(time_mean_ral.shape[0]), time_mean_ral - time_std_ral, time_mean_ral + time_std_ral, alpha=0.3)
ax.set_xlabel('Period')
ax.set_ylabel('Comp. Time')
ax.set_title('Time comparison')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# robotHistory = np.load(path / "robotHistory.npy")
# rmseHistory = np.load(path / "rmseHistory.npy")
# nlpdHistory = np.load(path / "nlpdHistory.npy")
# observedHistory = np.load(path / "observedHistory.npy")
# filteredHistory = np.load(path / "filteredHistory.npy")
# DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
# DAC_time_hist = np.load(path / "DAC_time_hist.npy")
# timeHistory = np.load(path / "timeHistory.npy")
# meanHistory = np.load(path / "meanHistory.npy")
# stdHistory = np.load(path / "stdHistory.npy")

# PERIOD = robotHistory.shape[2]
# coverageHistory = np.empty((PERIOD, 1))

# for t in range(PERIOD):
#     print(f"Time: {t}")
#     coverageHistory[t] = utils.coveragePerformanceFuncDataset(robotHistory[:, :, t], field, res=10, BBOX=BBOX)

# Plot the data
# Only sim 0



# ax.legend()

# # Plot the RMSE
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(rmseHistory[i, :], label=f'Robot {i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('RMSE')
# ax.set_title('RMSE')
# ax.legend()

# # Plot the NLPD
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(nlpdHistory[i, :], label=f'Robot {i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('NLPD')
# ax.set_title('NLPD')
# ax.legend()

# # Plot the coverage performance function
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(coverageHistory, label='Coverage')
# ax.set_xlabel('Time')
# ax.set_ylabel('Coverage')
# ax.set_title('Coverage')
# ax.legend()

# # Plot the observed and filtered data
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(observedHistory[i, :], label=f'Robot {i} observed', alpha=0.5, c=f'C{i}')
# for i in range(robNum):
#     ax.plot(filteredHistory[i, :], label=f'Robot {i} filtered', linestyle='--',  c=f'C{i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('Data')
# ax.set_title('Observed and filtered data')
# ax.legend()

# # Plot the DEC gapx time history
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(DEC_gapx_time_hist, label='DEC gapx')
# # Fill the area under the curve
# ax.fill_between(np.arange(PERIOD), DEC_gapx_time_hist[:, 0], alpha=0.3)
# ax.plot(DAC_time_hist, label='DAC')
# ax.fill_between(np.arange(PERIOD), DAC_time_hist[:, 0], alpha=0.5)
# ax.plot(timeHistory, label='Time')
# ax.fill_between(np.arange(PERIOD), timeHistory[:, 0], alpha=0.2)
# ax.set_xlabel('Time')
# ax.set_ylabel('Time')
# ax.set_title('Time history')
# ax.legend()

# plt.show()
