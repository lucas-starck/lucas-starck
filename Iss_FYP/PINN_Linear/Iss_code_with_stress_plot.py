#%%
#Imports
import numpy as np
import tensorflow as tf
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

tf.config.run_functions_eagerly(True)

#%%
#Importing data from Excel
df = pd.read_excel('Abaqus_data.xlsx', 0, header=0)


#Input data
N= 864  #no. of elements in each specimen - i.e. batch size
Exx_data = df.iloc[:, 1].values.astype('float32') # Exx size: (9504,1)
Eyy_data = df.iloc[:, 3].values.astype('float32') # Eyy size: (9504,1)
Exy_data = df.iloc[:, 5].values.astype('float32') # Exy size: (9504,1)
P_data = df.iloc[0:10, 12].astype('float32')   # P size: (11,1)
Ux_data = df.iloc[0:10, 11].astype('float32')  # Ux size: (11,1)
element_volume = df.iloc[0:N, 8].values.astype('float32') # Volume of each element for volume integral size: (9504,1)
x_coords = df.iloc[:, 25].values.astype('float32')
y_coords = df.iloc[:, 26].values.astype('float32')

P_data*=1e3
#Combining the features into one array
inputs = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1) #size: (9504, 3)
max_inputs = tf.reduce_max(inputs, axis=0)
min_inputs = tf.reduce_min(inputs, axis=0)
scaled_inputs = (inputs-min_inputs)/(max_inputs-min_inputs)
standardisation_params = tf.Variable(tf.zeros((2,3)), trainable=False)
standardisation_params.assign_add([max_inputs, min_inputs])
#print('shape of stand_params:', standardisation_params.shape)



#calculating external work
external_work =0.5* (P_data * Ux_data) # gives a vector of external work values at each time step/batch size: (11, 1)

print(external_work)

#Designing custom loss function

#%%
def custom_loss_with_params(external_work, standardisation_params, element_volume):
    def custom_loss(inputs, L):   

        #unscaling the inputs
        print('standardisation:', standardisation_params)
        max_inputs, min_inputs = tf.split(standardisation_params, 2)

        reversed_inputs = inputs * (max_inputs - min_inputs) + min_inputs

        #undoing the inputs normalisation

 
        array_inputs=np.array(reversed_inputs)
        
        Exx = array_inputs[:,0]
        Eyy = array_inputs[:,1]
        Exy = array_inputs[:,2]
        #vol = array_inputs[:,3] 
        
        vol_tf=tf.convert_to_tensor(element_volume, dtype=tf.float32)
        #epsilon is a combination of Exx, Eyy and Exy features
        epsilon = tf.stack([Exx, Eyy, Exy], axis=1) # size: (N, 3)
        epsilon = tf.reshape(epsilon , (epsilon.shape[0],3,1)) #sise:
        #print('epsilon:', epsilon)
        
        # generate epsilonT
        epsilonT = tf.transpose(epsilon,perm=[0,2,1]) # size: (3, N)
        #print('EpsilonT:',epsilonT)
        
        n=864. #no. of elements in the strain field
    
        #initialise loss function components
        strain_energy= 0.

    
        # L transpose
        LT = tf.transpose(L,perm=[0,2,1])
        #print(L_matrixT.shape)
        
        # Calculate C
        C = tf.matmul(L, LT) # size: (N,3,3) 
        #print('C', C)
        #print('C shape', C.shape)
    
        #External work seems to be a tf. tensor and strain energy is a Keras tensor - not sure if this matters?
    
        # Calculation of strain energy - matmul functions perform the matrix multiplication E_T xCx E, dot multiply by element volume vector, outputting a scalar
        temp1 = tf.matmul(epsilonT, tf.matmul(C, epsilon))
        #print('temp1', temp1)
        strain_energy = 0.5 * tf.tensordot(vol_tf , temp1 ,1)/n # do i need to times this by a half? # size is a scalar
        print('strain energy value:', strain_energy)
        
        difference = tf.abs((external_work[batch_no])-((strain_energy)))# calculates the squared difference between virtual work at the current batch and the strain energy - this may not be right

       
        #print('external work:',external_work.shape)
        print('external work value:', external_work[batch_no]) # doesn't seem to be outputting the correct value. 
        
        #mean_squared_difference = tf.reduce_mean(squared_difference, axis=-1) #Averages squared difference across each batch? Not sure this is necessary actually
        
        #rmse = tf.sqrt(squared_difference)
        print('difference RESULT: ',difference)
        return difference

    return custom_loss


#%%
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

model = Create_Model('linear', 2,20,20)
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), 
          loss = custom_loss_with_params(external_work=external_work, standardisation_params=standardisation_params, element_volume = element_volume)) #how do i make a custom loss function here?



#%%
#Train the model - what do i put in the place of train labels?

'''
history = model.fit(inputs, inputs, 
                    batch_size = N,
                    shuffle=False,
                    epochs = 100, 
                    verbose = 1) # haven't done validation data split yet
#validation_data = (valid_values, valid_labels),
'''

#training using train_on_batch
losses = []
L_batch_list=[]
batch_size = N
epochs = 500

# Initialize max and min values
#max_values = np.zeros((9,))
#min_values = np.zeros((9,))

for epoch in range(epochs):

    total_loss = 0.0
    batches_per_epoch = scaled_inputs.shape[0] // batch_size
    print('epoch number', epoch)
    custom_loss_calls = tf.Variable(0, trainable=False, dtype=tf.int32) # intialising counter for number of times loss function is called
    epoch_L_batch_list = []

    for i in range(0, len(scaled_inputs), batch_size):
        batch_no = custom_loss_calls.numpy()
        print('batch number',batch_no)
    
        x_batch = scaled_inputs[i:i+batch_size]
        
        #performing feature scaling per batch instead of on whole dataset
      
        y_batch = x_batch

        custom_loss_calls.assign_add(1) # adds one to the counter for how many times this function is entered/ which batch we are on
        # Train on the batch
        loss = model.train_on_batch(x_batch, y_batch, reset_metrics=True)
        L_batch = model.predict(x_batch)
        print('size_l_batch', L_batch.shape)
        epoch_L_batch_list.append(L_batch)
        

        #np.append(L_batch_list,L_batch, axis=0)

        total_loss += loss

    L_batch_list.append(np.array(epoch_L_batch_list))
    
    average_loss = tf.sqrt(tf.square(total_loss)/ batches_per_epoch)
    print('loss per batch', average_loss)
    losses.append(average_loss)

print('final loss', average_loss)

L_batch_array = np.concatenate(L_batch_list, axis=0)
last_epoch_values = L_batch_array[-10:]
reshaped_L = last_epoch_values.reshape((8640, 3, 3))
reshaped_L_tensor = tf.convert_to_tensor(reshaped_L, dtype=tf.float32)



print('reshapedL size:', np.shape(reshaped_L))
#getting output for last batch only (last time step)
    
L_finalT = tf.transpose(reshaped_L_tensor,perm=[0,2,1])

############################################################################################################################        
# Calculate C
C_final = tf.matmul(reshaped_L_tensor, L_finalT) # size: (N,3,3) 


print('final C', C_final)
print('final C shape', C_final.shape)



epsilon = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1)
epsilon = tf.reshape(epsilon , (epsilon.shape[0],3,1)) #sise:


stress = tf.matmul(C_final, epsilon)
#print('stress shape:', stress.shape)

reshaped_tensor = tf.reshape(stress, (stress.shape[0], -1))

numpy_array = reshaped_tensor.numpy()

# Specify the file path for the CSV file
csv_file_path = 'output_stress.csv'

# Use NumPy's savetxt function to save the array to a CSV file
np.savetxt(csv_file_path, numpy_array, delimiter=',')

# After training loop, you can use max_values and min_values
#print("Max values:", max_values)
#print("Min values:", min_values)


#loss = model.evaluate(inputs, inputs, verbose=0)

# Plot history
epochs_range  = np.arange(1, epochs + 1)
plt.plot(epochs_range, losses, label='Training Loss')
#plt.plot  ( epochs, history.history['val_loss' ], label = 'Validation')
#plt.title ('Training and validation loss')
#plt.legend();

#%%
##trying to plot a contour plot

def hole_mask(x, y,center, radius):
    return (x-center[0])**2 + (y- center[1])**2 < radius**2

# ... Your previous code to generate x_grid, y_grid, and stress ...


x_grid = np.linspace(-50, 50, num=1000)  # Adjust as needed
y_grid = np.linspace(-20, 20, num=400)    # Adjust as needed
x_grid, y_grid = np.meshgrid(x_grid, y_grid)

s11 = stress[7776:8640, 0]
#s11 /=1e6
stress_tensor = tf.convert_to_tensor(s11, dtype=tf.float32)
stress_1d = tf.reshape(stress_tensor, [-1])

print(stress_1d.shape)
print(x_coords[0:864].shape)
print(stress_1d)
      

grid_s11 = griddata((x_coords[0:864], y_coords[0:864]), stress_1d, (x_grid, y_grid), method='linear')


# Apply hole mask
hole_center = (np.mean(x_coords[0:864]), np.mean(y_coords[0:864])) # Adjust as needed
hole_radius = 2  # Adjust as needed
hole_mask_values = hole_mask(x_grid, y_grid, hole_center, hole_radius)
grid_s11[hole_mask_values] = np.nan  # Set values inside the hole to NaN

# Create a contour plot
plt.contourf(x_grid, y_grid, grid_s11, cmap='jet', levels=100)  # You can change cmap and levels as needed
plt.colorbar(label='Stress Component 11 Predicted Batch 10')

plt.xlim(-50, 50)
plt.ylim(-20, 20)
# Set labels and title
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.title('Stress Component 11 Contour Plot')

plt.gca().set_aspect('equal', adjustable='box')


# Show the plot
plt.show()

#plotting corresponding fea data field

df = pd.read_excel('stress_output_fea.xlsx', 0, header=0)


#Input data
N= 864  #no. of elements in each specimen - i.e. batch size
S11_fea = df.iloc[:, 1].values.astype('float32') # Exx size: (9504,1)
stress_fea_tensor = tf.convert_to_tensor(S11_fea, dtype=tf.float32)
#stress_1d = tf.reshape(stress_tensor, [-1])


grid_s11_fea = griddata((x_coords[0:864], y_coords[0:864]), stress_fea_tensor, (x_grid, y_grid), method='linear')


# Apply hole mask
grid_s11_fea[hole_mask_values] = np.nan  # Set values inside the hole to NaN

# Create a contour plot
plt.contourf(x_grid, y_grid, grid_s11_fea, cmap='jet', levels=100)  # You can change cmap and levels as needed
plt.colorbar(label='Stress Component 11')

plt.xlim(-50, 50)
plt.ylim(-20, 20)
# Set labels and title
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.title('Stress Component 11 Contour Plot FEA Batch 10')

plt.gca().set_aspect('equal', adjustable='box')


# Show the plot
plt.show()
