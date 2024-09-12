import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# Load data
path = Path('')
dataset_path = path / "dataset.nc"

ds = xr.open_dataset(dataset_path)
sst = ds['analysed_sst']

# Plot the region of interest (California)
lat_bounds = slice(22, 23)
lon_bounds = slice(-115, -114)

# Select the data within the region
sst_region = sst.sel(lat=lat_bounds, lon=lon_bounds)

# To NumPy array
sst_region = sst_region.values

# From Kelvin to Celsius
sst_region = sst_region - 273.15

# From 3D to 2D
sst_region = sst_region[0, :, :]

# Convert to float32 for memory efficiency
sst_region = sst_region.astype(np.float32)

# Define the original grid
original_x = np.linspace(0, sst_region.shape[1], sst_region.shape[1])
original_y = np.linspace(0, sst_region.shape[0], sst_region.shape[1])

# Define the target grid (64x64)
target_size = (100, 100)
target_x = np.linspace(0, sst_region.shape[1], target_size[1])
target_y = np.linspace(0, sst_region.shape[0], target_size[0])

# Create meshgrid for original and target coordinates
original_x_grid, original_y_grid = np.meshgrid(original_x, original_y)
target_x_grid, target_y_grid = np.meshgrid(target_x, target_y)

# Flatten the grids and SST data for interpolation
original_coords = np.vstack([original_x_grid.flatten(), original_y_grid.flatten()]).T
target_coords = np.vstack([target_x_grid.flatten(), target_y_grid.flatten()]).T
sst_flat = sst_region.flatten()

# Perform interpolation
sst_interpolated = griddata(original_coords, sst_flat, target_coords, method='linear')
sst_interpolated = sst_interpolated.reshape(target_size)

# Min-Max Normalization
sst_min = np.min(sst_interpolated)
sst_max = np.max(sst_interpolated)
sst_normalized = (sst_interpolated - sst_min) / (sst_max - sst_min)

# Define the custom point of interest in the new grid
custom_point_of_interest = (5.5, 5.5)  # Adjusted to fit within the new grid

# Create a grid for the interpolated SST data
interpolated_x_grid = np.linspace(0, target_size[1]-1, target_size[1])
interpolated_y_grid = np.linspace(0, target_size[0]-1, target_size[0])
interpolated_x_grid, interpolated_y_grid = np.meshgrid(interpolated_x_grid, interpolated_y_grid)

# Flatten the interpolated grid and SST data
interpolated_coords = np.vstack([interpolated_x_grid.flatten(), interpolated_y_grid.flatten()]).T
sst_interpolated_flat = sst_normalized.flatten()

# Perform interpolation to find the value at the custom point
sst_at_custom_point = griddata(interpolated_coords, sst_interpolated_flat, [custom_point_of_interest], method='linear')

print(f"SST at custom point {custom_point_of_interest}: {sst_at_custom_point[0]}")

# Plot the region of interest
plt.figure(figsize=(10, 5))
plt.imshow(sst_normalized, cmap='jet', origin='lower')
plt.scatter(custom_point_of_interest[0], custom_point_of_interest[1], c='red', s=100, marker='x')
plt.colorbar()
plt.title('SST')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()