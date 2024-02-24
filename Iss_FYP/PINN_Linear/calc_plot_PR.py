import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import tensorflow as tf


def calc_plot_PR(x_coords, y_coords, C_final, N, true_PR = 0.34):


    C_last_batch = C_final[-N:, :, :]


    trace_C = np.trace(C_last_batch, axis1=1, axis2=2)


    # Young's Modulus
    poisson_ratio_pred = C_last_batch[:, 2, 1] / C_last_batch[:, 1, 1]


    #percentage error:

    error_PR = (tf.abs(poisson_ratio_pred-true_PR)/true_PR) * 100

    x_grid = np.linspace(-50, 50, num=1000)  #L=100mm
    y_grid = np.linspace(-20, 20, num=400)    #W=40mm
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    #function to add the hole to the contour plot and prevent it being filled
    def hole_mask(x, y,center, radius):
        return (x-center[0])**2 + (y- center[1])**2 < radius**2

    # Apply hole mask
    hole_center = (np.mean(x_coords), np.mean(y_coords)) # Adjust as needed
    hole_radius = 2  # Adjust as needed
    hole_mask_values = hole_mask(x_grid, y_grid, hole_center, hole_radius)

    #contour plots of E and v percentage difference 

    grid_PR = griddata((x_coords, y_coords), error_PR, (x_grid, y_grid), method='linear')


    grid_PR[hole_mask_values] = np.nan


    cbar3 = plt.contourf(x_grid, y_grid, grid_PR, cmap='jet', levels=100)  # You can change cmap and levels as needed
    cbar3 = plt.colorbar(label='Percentage Error in PoissonRatio')
    plt.xlim(-50, 50)
    plt.ylim(-20, 20)
    # Set labels and title
    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Y Coordinate (mm)')
    plt.title('Poisson Ratio Error Field')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    #print("Poisson's ratio error :", error_PR)

    return error_PR, poisson_ratio_pred