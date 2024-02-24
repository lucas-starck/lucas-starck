import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_stress_contours(x_coords, y_coords, stress, N):

    #trying to plot a contour plot of the final stress batch prediction

    #altering x coords
    transform_x = abs(50.0 - max(x_coords))
    x_coords = [coord - transform_x for coord in x_coords]
    transform_y = abs(20.0 - max(y_coords))
    y_coords = [coord + transform_y for coord in y_coords]


    #function to add the hole to the contour plot and prevent it being filled
    def hole_mask(x, y,center, radius):
        return (x-center[0])**2 + (y- center[1])**2 < radius**2


    #creating a grid for the countour plot
    x_grid = np.linspace(-50, 50, num=1000)  #L=100mm
    y_grid = np.linspace(-20, 20, num=400)    #W=40mm
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    #extracting final batch stress prediction
    s11 = stress[:N, 0]  # final batch
    s22 = stress[:N, 1] 
    s12 = stress[:N:, 2] 
    #s11 /=1e6

    #converting the stress to a tensor and reshaping to a 1d array
    s11_tensor = tf.convert_to_tensor(s11, dtype=tf.float32)
    s11_1d = tf.reshape(s11_tensor, [-1])
    s22_tensor = tf.convert_to_tensor(s22, dtype=tf.float32)
    s22_1d = tf.reshape(s22_tensor, [-1])
    s12_tensor = tf.convert_to_tensor(s12, dtype=tf.float32)
    s12_1d = tf.reshape(s12_tensor, [-1])

    #print('shape xcoords:', len(x_coords))
    #generating grid using FEA x and y coords and batch 10 predicted stress field
    grid_s11 = griddata((x_coords, y_coords), s11_1d, (x_grid, y_grid), method='linear')
    grid_s22 = griddata((x_coords, y_coords), s22_1d, (x_grid, y_grid), method='linear')
    grid_s12 = griddata((x_coords, y_coords), s12_1d, (x_grid, y_grid), method='linear')

    # Apply hole mask
    hole_center = (np.mean(x_coords), np.mean(y_coords)) # Adjust as needed
    hole_radius = 2  # Adjust as needed
    hole_mask_values = hole_mask(x_grid, y_grid, hole_center, hole_radius)

    grid_s11[hole_mask_values] = np.nan  # Set values inside the hole to NaN
    grid_s22[hole_mask_values] = np.nan  # Set values inside the hole to NaN
    grid_s12[hole_mask_values] = np.nan  # Set values inside the hole to NaN

    # Create S11 contour plot

    cbar1 = plt.contourf(x_grid, y_grid, grid_s11, cmap='jet', levels=100)  # You can change cmap and levels as needed
    cbar1 = plt.colorbar(label='Stress Component 11 Predicted Batch 10')
    plt.xlim(-50, 50)
    plt.ylim(-20, 20)
    # Set labels and title
    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Y Coordinate (mm)')
    plt.title('Stress Component 11 Contour Plot')
    plt.gca().set_aspect('equal', adjustable='box')
    # Show the plot
    plt.show()

    # Create S22 Contour plot
    # Vmax = 1.0e9 # max colourbar value
    # Vmin = 0. # min colourbar value
    # cbar2 = plt.contourf(x_grid, y_grid, grid_s22, cmap='jet', levels=100, vmin=Vmin,vmax=Vmax)  # You can change cmap and levels as needed
    # cbar2 = plt.colorbar(label='Stress Component 22 Predicted Batch 10', 
    #              ticks=[x / 10. for x in range(int(Vmin),int(Vmax*10)+1)] ,
    #              extend='both')
    # plt.xlim(-50, 50)
    # plt.ylim(-20, 20)
    # # Set labels and title
    # plt.xlabel('X Coordinate (mm)')
    # plt.ylabel('Y Coordinate (mm)')
    # plt.title('Stress Component 22 Contour Plot')
    # plt.gca().set_aspect('equal', adjustable='box')
    # # Show the plot
    # plt.show()

    # Create S12 Contour plot
    # Vmax = 1.0e9 # max colourbar value
    # Vmin = 0. # min colourbar value
    # cbar3 = plt.contourf(x_grid, y_grid, grid_s12, cmap='jet', levels=100, vmin=Vmin,vmax=Vmax)  # You can change cmap and levels as needed
    # cbar3 = plt.colorbar(label='Stress Component 12 Predicted Batch 10', 
    #              ticks=[x / 10. for x in range(int(Vmin),int(Vmax*10)+1)] ,
    #              extend='both')
    # plt.xlim(-50, 50)
    # plt.ylim(-20, 20)
    # # Set labels and title
    # plt.xlabel('X Coordinate (mm)')
    # plt.ylabel('Y Coordinate (mm)')
    # plt.title('Stress Component 12 Contour Plot')
    # plt.gca().set_aspect('equal', adjustable='box')
    # # Show the plot
    # plt.show()

    return