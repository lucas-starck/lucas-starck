#%%##########################################################################################################################
#Imports
import numpy as np
import tensorflow as tf
import os, sys, time
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import timeit
from glob import glob
from matplotlib.ticker import MaxNLocator
tf.config.run_functions_eagerly(True)



#%%###########################################################################################################################
# Extracting FEA stress output

#importing coordinates and number of elements
root_file_path = '/Users/lucasastarck/Documents/Data Science/Coding/lucas-starck/Iss_FYP/PINN_Linear/FEA_data'
inp_file_path = root_file_path+'/Plate_hole_1.inp'
target_section = '*Node'
columns_to_load = [1, 2]

with open(inp_file_path, 'r') as inp_file:
    in_target_section = False
    nodal_data = []

    for line in inp_file:
        # Check if the line contains the target section header
        if target_section in line:
            in_target_section = True
            continue

        # If in the target section, read and process the lines until the next section
        if in_target_section:
            # Check if the line starts with an asterisk, indicating the start of a new section
            if line.startswith('*'):
                break

            # Split the line into fields and select only the specified columns
            fields = line.strip().split(',')
            selected_columns = [float(fields[i]) for i in columns_to_load]

            # Store the selected columns in nodal_data
            nodal_data.append(selected_columns)

x_coords = [entry[0] for entry in nodal_data]
y_coords = [entry[1] for entry in nodal_data]


no_batches = 10
N=len(x_coords)

#outputting stress and strain data from fea
data_list_strain = []
data_list_stress = []

for file_idx in range(1, no_batches+1):
    # Load text file using numpy.loadtxt
    filename_strain = f'strain_output_frame_{file_idx}.txt'
    filename_stress = f'stress_output_frame_{file_idx}.txt'

    file_path_strain = glob(os.path.join(root_file_path, filename_strain))
    if file_path_strain:
            # Load text file using numpy.loadtxt
            data_strain = np.loadtxt(file_path_strain[0], dtype='float32', usecols=(1, 3, 5), skiprows=23, max_rows=N)
            
            # Append the data to the list
            data_list_strain.append(data_strain)
    else:
        print(f"File {filename_strain} not found.")

    file_path_stress = glob(os.path.join(root_file_path, filename_stress))
    if file_path_stress:
            # Load text file using numpy.loadtxt
            data_stress = np.loadtxt(file_path_stress[0], dtype='float32', usecols=(1, 3, 5), skiprows=23, max_rows=N)
            
            # Append the data to the list
            data_list_stress.append(data_stress)
    else:
        print(f"File {filename_stress} not found.")

# Concatenate data from all files
final_data_strain = np.concatenate(data_list_strain, axis=0)
final_data_stress = np.concatenate(data_list_stress, axis=0)

Exx_data = np.exp(final_data_strain[:, 0]) - 1
Eyy_data = np.exp(final_data_strain[:, 1]) - 1
Exy_data = np.exp(final_data_strain[:, 2]) - 1

print((Exx_data).shape)

Sxx_data = final_data_stress[:, 0]
Syy_data = final_data_stress[:, 1]
Sxy_data = final_data_stress[:, 2]


#outputting volume and displacement/force data from fea

filename_vol = 'volume.txt'
file_path_vol = glob(os.path.join(root_file_path, filename_vol))
element_vol = np.loadtxt(file_path_vol[0], dtype='float32', usecols=(1), skiprows=22, max_rows=N)


filename_ext_work = 'force_displacement_data.txt'
file_path_ext_work = glob(os.path.join(root_file_path, filename_ext_work))
data_ext_work = np.loadtxt(file_path_ext_work[0], dtype='float32', usecols=(0, 2), skiprows=4, max_rows=N)

Ux_data = data_ext_work[1:,0] 
P_data = data_ext_work[1:,1] 


#P_data*=1e3 # trying to put the unit N into mm

# Extracting scaling values to scale between (0,1) and storing them
#   Note: we are scaling by the max/min value of all 10 batches, and not of each batch
inputs = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1) #size: (9504, 3)
max_inputs = tf.reduce_max(inputs, axis=0)
min_inputs = tf.reduce_min(inputs, axis=0)
normalisation_params = tf.Variable(tf.zeros((2,3)), trainable=False)
normalisation_params.assign_add([max_inputs, min_inputs])

scaled_inputs = (inputs-min_inputs)/(max_inputs-min_inputs)

#calculating external work
external_work =0.5* (P_data * Ux_data) # gives a vector of external work values at each time step/batch size: (11, 1)
print('External work: '); print(external_work); print('')


#%%##########################################################################################################################
#Designing custom loss function
#Wrapper function
def custom_loss_with_params(external_work, normalisation_params, element_vol, N_per_batch):

    def custom_loss(inputs_scaled, L):   

        # Un-scaling inputs for physical calculations
        max_inputs, min_inputs = tf.split(normalisation_params, 2)
        inputs_unscaled = inputs_scaled * (max_inputs - min_inputs) + min_inputs

        #convert volume array to a tensor
        vol=tf.convert_to_tensor(element_vol, dtype=tf.float32)
    
        # Initialise total energy difference vector
        Energy_difference_epoch = []

        #iterate across the batches to get RMSE energy difference
        for batch_idx in range(0, len(inputs_unscaled), N_per_batch): # (0, Nx10, N)

            # Define batch data
            input_batch = inputs_unscaled[batch_idx:batch_idx+N_per_batch] # generates 1 batch of size N
            L_batch = L[batch_idx:batch_idx+N_per_batch]
            strain_energy_batch = 0.
            n = float(N_per_batch) # no. elements
            # convert batch index to batch number
            if batch_idx == 0: # account for zero division
                batch_number = 0 
            else:
                batch_number = int(batch_idx/N_per_batch) 
                if batch_idx % N_per_batch != 0: # check that epoch size is divisible by batch size
                    raise ValueError('The number of elements in the epoch',len(inputs_unscaled),'must be divisible by the number of elements in the batch',N_per_batch)
            print('Batch number: ',batch_number)

            # Separating input data into variables
            array_inputs=np.array(input_batch) # do I need this?
            Exx_batch = array_inputs[:,0]
            Eyy_batch = array_inputs[:,1]
            Exy_batch = array_inputs[:,2]

            # Calculate epsilon and its transpose (for the batch)
            epsilon_batch = tf.stack([Exx_batch, Eyy_batch, Exy_batch], axis=1) # size: (N, 3)
            epsilon_batch = tf.reshape(epsilon_batch , (epsilon_batch.shape[0],3,1))
            epsilonT_batch = tf.transpose(epsilon_batch,perm=[0,2,1]) # size: (3, N)

            # Calculate L transpose from input L
            LT_batch = tf.transpose(L_batch,perm=[0,2,1])
        
            # Calculate C = L*LT
            C_batch = tf.matmul(L_batch, LT_batch) # size: (N,3,3) 

            # Calculate strain energy for batch: 
            #   Matrix multiply E_T * C * E, and dot multiply by element volume, outputting a scalar
            strain_energy_batch = 0.5 * tf.tensordot(vol , tf.matmul(epsilonT_batch, tf.matmul(C_batch, epsilon_batch) ) ,1) / n # scalar value
            print('Internal work (strain energy) - batch',batch_number,':', strain_energy_batch[0][0].numpy())
            print('External work (F x d)         - batch',batch_number,':', external_work[batch_number]) 
                
            # Calculate Loss for batch: 
            #   Difference between internal (strain energy) and external work (F*d)
            Energy_difference_batch = tf.abs((external_work[batch_number])-((strain_energy_batch)))
            print('Energy difference             - batch',batch_number,':',Energy_difference_batch[0][0].numpy()); print('') # print newline

            # Add batch energy difference to total epoch energy difference
            # Accumulate Energy_difference_batch to the array
            Energy_difference_epoch.append(Energy_difference_batch)
            Energy_difference_epoch_total = tf.concat([Energy_difference_epoch], axis=0)
            
            
        print(Energy_difference_epoch_total.shape)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(Energy_difference_epoch_total)))
        print('rmse:', rmse); print(' ') # print newline

        return rmse     

    return custom_loss


#%%###############################################################################################################
# Creating ML Model

def Create_Model(activ, Hidden_layers, node_num1, node_num2):
    """
    Creates a Keras sequential model with the specified number of hidden layers and nodes, 
    with the given activation function for all layers except the output layer, which uses a linear activation.
    
    Args:
        activ (str): the name of the activation function to use in all hidden layers
        Hidden_layers (int): the number of hidden layers in the model with node_num2 nodes each. 
                            Total_hidden_layers = Hidden_layers + 1
        node_num1 (int): the number of nodes in the first hidden layer
        node_num2 (int): the number of nodes in each subsequent hidden layer
        
    Returns:
        model (tf.keras.Sequential): a Keras sequential model with the specified architecture
    """
    
    model=tf.keras.Sequential()

    # Add the first hidden layer with the specified number of nodes and activation function
    # Input shape is set to [4] to match the shape of the input data
    model.add(tf.keras.layers.Dense(node_num1, activation=activ, input_shape=[3], name='hidden_layer_0'))

    #model.add(tf.keras.layers.BatchNormalization()) # performs normalisation on the input data, and normalises the output during training 
    
    # Add additional hidden layers with the same number of nodes and activation function
    for j in range(Hidden_layers):
        #model.add(tf.keras.layers.Dropout(0.2))  #regularisation
        model.add(tf.keras.layers.Dense(node_num2, activation=activ, name='hidden_layer_'+str(j+1)))
    
    # Add the output layer with linear activation and 9 nodes to match the desired output shape
    model.add(tf.keras.layers.Dense(9, activation='linear', name='output_layer'))

    model.add(tf.keras.layers.Reshape((3, 3), input_shape=(9,))) # reshapes the output into a 3x3 tensor
    
    # Gets the output from the output layer to then pass into the custom loss function
    
    # Return the fully-defined model
    return model


#%%
#Calling create model amd compiling model with optimizer

model = Create_Model('linear', Hidden_layers=2, node_num1=20, node_num2=20)
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), 
          loss = custom_loss_with_params(external_work = external_work, 
                                         normalisation_params = normalisation_params,
                                         element_vol = element_vol,
                                         N_per_batch = N))



#%%###################################################################################################
# Train ML Model manually using train_on_batch


epoch_loss_list = []
epoch_time_list = []
epoch_time_cumsum = 0.
L_batch_list=[]
batches_per_epoch = scaled_inputs.shape[0] // N
batch_size = N
N_epochs = 500 #500 will converge

# Initialize max and min values
#max_values = np.zeros((9,))
#min_values = np.zeros((9,))

for epoch in range(N_epochs):

    L_epoch=[]

    # Start timer to record how long each epoch takes to run
    epoch_start = timeit.default_timer() 

    total_loss = 0.0 # sum of loss of all batches in this epoch
    print('Epoch number: ', epoch) ; print('')

    # Train all 10 timesteps at once (one batch of 8640 records)
    loss = model.train_on_batch(x=scaled_inputs, y=scaled_inputs)
    
    # Track loss and time of all epochs for plotting
    epoch_loss_list.append(loss) 
    epoch_time_cumsum += float(timeit.default_timer() - epoch_start)
    epoch_time_list.append(epoch_time_cumsum)


    # Break training if runtime limit is exceeded
    if epoch_time_cumsum > 200.:
        break
  

#predciting L at each epoch - overwrites but that's okay because i only want last epoch
L_epoch = model.predict(scaled_inputs)

print('shape of L predicted:', L_epoch.shape)

#Calculates L and LT as tensors
L_epoch_tensor = tf.convert_to_tensor(L_epoch, dtype=tf.float32)# size: (8640,3,3) 
L_finalT = tf.transpose(L_epoch_tensor,perm=[0,2,1])# size: (8640,3,3) 
    
# Calculate C prediction (at every example)
C_final = tf.matmul(L_epoch_tensor, L_finalT) # size: (8640,3,3) 

print('C final:', C_final)
print('C final shape:', C_final.shape)

#Getting unscaled input data
epsilon = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1)
epsilon = tf.reshape(epsilon , (epsilon.shape[0],3,1)) #size:(8640,3,1) 

#Calculate the stress stress=Cxepsilon
stress = tf.matmul(C_final, epsilon) 

#reshaping stress for output
reshaped_tensor = tf.reshape(stress, (stress.shape[0], -1))
numpy_array = reshaped_tensor.numpy()

# Specify the file path for the CSV file
csv_file_path = 'output_stress.csv'

# Use NumPy's savetxt function to save the stress array to a CSV file
np.savetxt(csv_file_path, numpy_array, delimiter=',')


#%%###############################################################################################################################
#plot training loss history

fig = plt.figure()

# Loss vs Epochs subplot
fig1 = fig.add_subplot(2,1,1)
N_epochs_range  = np.arange(1, len(epoch_loss_list) + 1)
fig1.plot(N_epochs_range, epoch_loss_list)
fig1.grid()
fig1.set_ylabel('Training loss')
fig1.set_xlabel('Epoch number')
fig1.set_title('Training loss history')
fig1.set_ylim((0,4e4))

print('epochs loss list:', epoch_loss_list)


# Loss vs Runtime subplot
fig2 = fig.add_subplot(2,1,2)
fig2.semilogy(epoch_time_list, epoch_loss_list)
fig2.grid()
fig2.set_ylabel('Training loss')
fig2.set_xlabel('Runtime')
fig2.set_ylim((0,4e4))
plt.tight_layout()


#%%################################################################################################################################
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

print('shape xcoords:', len(x_coords))
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


##########################################################################################
#plotting corresponding fea data field

#Input data
S11_fea = Sxx_data[:N]  #convert to Pa as fea outputs  Pa
s11_fea_tensor = tf.convert_to_tensor(S11_fea, dtype=tf.float32)
#Input data
S22_fea = Syy_data[:N]   #convert to Pa as fea outputs x10^3 Pa
s22_fea_tensor = tf.convert_to_tensor(S22_fea, dtype=tf.float32)
#Input data
S12_fea = Sxy_data[:N]  #convert to Pa as fea outputs x10^3 Pa
s12_fea_tensor = tf.convert_to_tensor(S12_fea, dtype=tf.float32)


grid_s11_fea = griddata((x_coords, y_coords), s11_fea_tensor, (x_grid, y_grid), method='linear')
grid_s22_fea = griddata((x_coords, y_coords), s22_fea_tensor, (x_grid, y_grid), method='linear')
grid_s12_fea = griddata((x_coords, y_coords), s12_fea_tensor, (x_grid, y_grid), method='linear')

# # Apply hole mask
grid_s11_fea[hole_mask_values] = np.nan  # Set values inside the hole to NaN
grid_s22_fea[hole_mask_values] = np.nan  # Set values inside the hole to NaN
grid_s12_fea[hole_mask_values] = np.nan  # Set values inside the hole to NaN

# Create S11 contour plot

cbar1 = plt.contourf(x_grid, y_grid, grid_s11_fea, cmap='jet', levels=100)  # You can change cmap and levels as needed
cbar1 = plt.colorbar(label='Stress Component 11')
plt.xlim(-50, 50)
plt.ylim(-20, 20)
# Set labels and title
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.title('Stress Component 11 Contour Plot FEA Batch 10')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Create S22 contour plot

# cbar2 = plt.contourf(x_grid, y_grid, grid_s22_fea, cmap='jet', levels=100)  # You can change cmap and levels as needed
# cbar2 = plt.colorbar(label='Stress Component 22 (Pa)')
# plt.xlim(-50, 50)
# plt.ylim(-20, 20)
# # Set labels and title
# plt.xlabel('X Coordinate (mm)')
# plt.ylabel('Y Coordinate (mm)')
# plt.title('Stress Component 22 Contour Plot FEA Batch 10')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# Create S12 contour plot
# Vmax = 1.0e9 # max colourbar value
# Vmin = 0. # min colourbar value
# cbar3 = plt.contourf(x_grid, y_grid, grid_s12_fea, cmap='jet', levels=100, vmin=Vmin,vmax=Vmax)  # You can change cmap and levels as needed
# cbar3 = plt.colorbar(label='Stress Component 12 (Pa)', 
#              ticks=[x / 10. for x in range(int(Vmin),int(Vmax*10)+1)] ,
#              extend='both')
# plt.xlim(-50, 50)
# plt.ylim(-20, 20)
# # Set labels and title
# plt.xlabel('X Coordinate (mm)')
# plt.ylabel('Y Coordinate (mm)')
# plt.title('Stress Component 12 Contour Plot FEA Batch 10')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()


# %%

#trying to output an estimate for E and poissons ratio
True_YM = 1.65e5
True_PR = 0.34

C_last_batch = C_final[-N:, :, :]


trace_C = np.trace(C_last_batch, axis1=1, axis2=2)

# Poisson's Ratio
poisson_ratio_pred = C_last_batch[:, 2, 1] / C_last_batch[:, 1, 1]

# Young's Modulus
youngs_modulus_pred = trace_C / (3 * (1 - 2 * poisson_ratio_pred))

print("Young's Modulus pred :", youngs_modulus_pred)
print("Poisson's Ratio pred :", poisson_ratio_pred)
#percentage error:

error_YM = (tf.abs(youngs_modulus_pred-True_YM)/True_YM) * 100
error_PR = (tf.abs(poisson_ratio_pred-True_PR)/True_PR) * 100


#contour plots of E and v percentage difference 

grid_YM = griddata((x_coords, y_coords), error_YM, (x_grid, y_grid), method='linear')
grid_PR = griddata((x_coords, y_coords), error_PR, (x_grid, y_grid), method='linear')

grid_YM[hole_mask_values] = np.nan
grid_PR[hole_mask_values] = np.nan

cbar3 = plt.contourf(x_grid, y_grid, grid_YM, cmap='jet', levels=100)  # You can change cmap and levels as needed
cbar3 = plt.colorbar(label='Percentage Error in Young Modulus')
plt.xlim(-50, 50)
plt.ylim(-20, 20)
# Set labels and title
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.title('Young Modulus Error Field')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

cbar4 = plt.contourf(x_grid, y_grid, grid_PR, cmap='jet', levels=100)  # You can change cmap and levels as needed
cbar4 = plt.colorbar(label='Percentage Error in Poisson Ratio')
plt.xlim(-50, 50)
plt.ylim(-20, 20)
# Set labels and title
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.title('Poisson Ratio Error Field')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


print("Young's Modulus error :", error_YM)
print("Poisson's Ratio error :", error_PR)

# %%
