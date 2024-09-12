import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
from robot_RAL import Robot
import time
from pathlib import Path
import xarray as xr
from scipy.interpolate import griddata
import concurrent.futures

# Generate data
np.random.seed(23)

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

# # Define the custom point of interest in the new grid
# custom_point_of_interest = [(5.5, 5.5), (36.8, 11.1)]  # Adjusted to fit within the new grid

# pnt = evaluate_points_in_field(field, [custom_point_of_interest], method='linear')

# # # Interpolate to find the value at the custom point
# # sst_at_custom_point = griddata(
# #     np.column_stack([g.flatten() for g in target_grid]),
# #     field.flatten(),
# #     [custom_point_of_interest],
# #     method='linear'
# # )

# # Plot the region of interest
# plt.figure(figsize=(10, 5))
# c = plt.contourf(np.linspace(0, target_size[1]-1, target_size[1]),
#              np.linspace(0, target_size[0]-1, target_size[0]),
#              field, cmap='YlOrRd')
# plt.scatter([p[0] for p in custom_point_of_interest], [p[1] for p in custom_point_of_interest], c='black', marker='x', label='Custom Point')
# plt.colorbar(c)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Sea Surface Temperature (SST)')
# plt.gca().set_aspect('equal')
# plt.legend()
# plt.show()


# # Generate random means
# peaks = 2 # np.random.randint(1, 10)
# means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
# sigma = 6
# Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
# Z = Z.reshape(len(x1_), len(x2_))

""" Robots parameters """
ROB_NUM = 6
CAMERA_BOX = 2
CAMERA_SAMPLES = 10

_area_to_cover = (x_sup * y_sup) * 1.0

RANGE = 2 * np.sqrt((_area_to_cover / ROB_NUM) / np.pi)

K_GAIN = 3
D_t = 0.1

robots = np.empty(ROB_NUM, dtype=object)

safety_dist = 5
for r in np.arange(ROB_NUM):
    x1, x2 = np.random.uniform(0 + safety_dist, (x_sup - safety_dist)), np.random.uniform(0 + safety_dist, (y_sup - safety_dist))
    rob = Robot(total_robots=ROB_NUM,
                id=r,
                x1_init=x1,
                x2_init=x2,
                x1Vals=x1_,
                x2Vals=x2_,
                sensing_range=RANGE,
                sensor_noise=0.1,
                bbox=BBOX,
                mesh=mesh,
                field_delta=d_field_)
    robots[r] = rob

""" Figures """

PERIOD = 150
index_init = 600
index_end = 1400
step = (index_end - index_init) // PERIOD
index = index_init

""" Hystories """
robotHistory = np.empty((ROB_NUM, 2, PERIOD)) # History of the robots' positions
nlpdHistory = np.empty((ROB_NUM, PERIOD)) # History of the robots' NLPD
rmseHistory = np.empty((ROB_NUM, PERIOD)) # History of the robots' RMSE
observedHistory = np.empty((ROB_NUM, PERIOD)) # History of the robots' observed points
filteredHistory = np.empty((ROB_NUM, PERIOD)) # History of the robots' filtered points
DEC_gapx_time_hist = np.empty((PERIOD, 1))
DAC_time_hist = np.empty((PERIOD, 1))
timeHistory = np.empty((PERIOD, 1))
meanHistory = np.empty((PERIOD, ROB_NUM, field.shape[0], field.shape[1])) # History of the mean of the robots
stdHistory = np.empty((PERIOD, ROB_NUM, field.shape[0], field.shape[1])) # History of the std of the robots

""" Network parameters """
A = np.zeros((ROB_NUM, ROB_NUM)) # Adjacency matrix

""" DEC-apx-GP """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3

""" DEC-PoE """
beta = 1 / ROB_NUM
s_end_DAC = 100

"""
Main loop
"""
for t in np.arange(0, PERIOD):
    print(f"\n=== Step: {t} ===")

    if t == 0:
        first = True
    else:
        first = False

    utils.sense_neighbors(robots)
    utils.update_A(robots, A)
    utils.find_groups(robots, A)
    groups = [robot.group for robot in robots]

    degrees = np.sum(A, axis=1)
    max_degree = np.max(degrees)

    eps = 1 / (max_degree)
    eps = eps / 2

    sTime = time.time()
    for i, robot in enumerate(robots):
        robot.time = t
        robot.compute_voronoi()

        # Take 5 random points from the robot sensing area
        points = np.random.uniform(robot.position - CAMERA_BOX, robot.position + CAMERA_BOX, (int(CAMERA_SAMPLES), 2))
        # y_values = utils.gmm_pdf_array(points[:, 0], points[:, 1], sigma, means, flag_normalize=False) + robot.sensor_noise * np.random.randn(len(points))
        y_values = utils.evaluate_points_in_field(field, points, method='linear') + robot.sensor_noise * np.random.randn(len(points))
        
        robot.sense(points, y_values, first=first)
        robot.update_dataset() # Update the dataset with the new observation

    for robot in robots:
        for other_robot in robot.neighbors:
            robot.sense(other_robot.observations[:, :2], other_robot.observations[:, 2], first=first)

    for robot in robots:
        robot.update_dataset()
        print(f"Robot {robot.id} has {robot.observations.shape[0]} observations")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(Robot.fit, robots)
        for result in results:
            pass

    for robot in robots:
        robot.update_estimate()

    for i, robot in enumerate(robots):
        robot.compute_centroid()

    # Move the robots
    for robot in robots:
        x1, x2 = robot.position + (-K_GAIN*(robot.position - robot.centroid) * D_t)
        robot.move(x1, x2)

    timeHistory[t] = time.time() - sTime

    """ Histories update """
    for i, robot in enumerate(robots):
        robotHistory[i, :, t] = robot.position
        # Compute RMSE between the real field and the robot's dataset
        rmseHistory[i, t] = np.sqrt(np.mean((field - robot.mean)**2))

        # Compute NLPD
        nlpd = 0.5 * np.log(2 * np.pi * robot.std**2) + (field - robot.mean)**2 / (2 * robot.std**2)
        nlpdHistory[i, t] = np.mean(nlpd)

        # Compute observed points
        observedHistory[i, t] = robot.observations.shape[0]

        # Compute filtered points
        filteredHistory[i, t] = robot.dataset.shape[0]

        meanHistory[t, i, :, :] = robot.mean

# Save the data
path = Path().resolve()
data_folder = path / "ral-sims/sim-3"

np.save(data_folder / "robotHistory.npy", robotHistory)
np.save(data_folder / "rmseHistory.npy", rmseHistory)
np.save(data_folder / "nlpdHistory.npy", nlpdHistory)
np.save(data_folder / "observedHistory.npy", observedHistory)
np.save(data_folder / "filteredHistory.npy", filteredHistory)
np.save(data_folder / "timeHistory.npy", timeHistory)
np.save(data_folder / "meanHistory.npy", meanHistory)
np.save(data_folder / "stdHistory.npy", stdHistory)
np.save(data_folder / "DEC_gapx_time_hist.npy", DEC_gapx_time_hist)
np.save(data_folder / "DAC_time_hist.npy", DAC_time_hist)

# Plot the data
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, aspect='equal')
ax.contourf(np.linspace(0, target_size[1]-1, target_size[1]),
             np.linspace(0, target_size[0]-1, target_size[0]),
             field, cmap='YlOrRd')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Robot paths')
for i, robot in enumerate(robots):
    ax.plot(robotHistory[i, 0, :], robotHistory[i, 1, :], label=f'Robot {i}')
    ax.scatter(robotHistory[i, 0, -1], robotHistory[i, 1, -1], marker='o')
ax.legend()

# Plot the RMSE
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
for i, robot in enumerate(robots):
    ax.plot(rmseHistory[i, :], label=f'Robot {i}')
ax.set_xlabel('Time')
ax.set_ylabel('RMSE')
ax.set_title('RMSE')
ax.legend()

# Plot the NLPD
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
for i, robot in enumerate(robots):
    ax.plot(nlpdHistory[i, :], label=f'Robot {i}')
ax.set_xlabel('Time')
ax.set_ylabel('NLPD')
ax.set_title('NLPD')
ax.legend()
plt.show()
