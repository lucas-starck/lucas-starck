#%%########################################################################################################
#Imports
import numpy as np
import tensorflow as tf
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)




#%%########################################################################################################
# Importing data from Excel
df = pd.read_excel('Abaqus_data.xlsx', 0, header=0)

# Extract input data
N = 864  # no. of elements in each specimen - i.e. batch size
Exx_data = df.iloc[:, 1].values.astype('float32') # Exx size: (8640,1)
Eyy_data = df.iloc[:, 3].values.astype('float32') # Eyy size: (8640,1)
Exy_data = df.iloc[:, 5].values.astype('float32') # Exy size: (8640,1)
element_vol = df.iloc[:N, 8].values.astype('float32') # Volume of each element for volume integral size: (8640,1)
P_data = df.iloc[0:10, 12].astype('float32')   # P size: (11,1)
Ux_data = df.iloc[0:10, 11].astype('float32')  # Ux size: (11,1)

# Combining features into one array
inputs = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1) # size: (8640, 3)

# Extracting scaling values to scale between (0,1) and storing them
#   Note: we are scaling by the max/min value of all 10 batches, and not of each batch
max_inputs = tf.reduce_max(inputs, axis=0)
min_inputs = tf.reduce_min(inputs, axis=0)
normalisation_params = tf.Variable(tf.zeros((2,3)), trainable=False)
normalisation_params.assign_add([max_inputs, min_inputs])
print('Normalisation parameters: '); print(normalisation_params.numpy()); print('')

# Scaling inputs
scaled_inputs = (inputs-min_inputs)/(max_inputs-min_inputs)

# Calculating external work
#   This gives a vector of external work values at each time step/batch size: (11, 1)
external_work = 0.5 * (P_data * Ux_data)
print('External work: '); print(external_work); print('')




#%%########################################################################################################
#Designing custom loss function

# Wrapper function - lets you input global variables into loss function
def custom_loss_with_params(external_work, normalisation_params, element_vol, N_per_batch): 

    # Loss function
    def custom_loss(inputs_scaled, L):  

        # Un-scaling inputs for physical calculations
        max_inputs, min_inputs = tf.split(normalisation_params, 2)
        inputs_unscaled = inputs_scaled * (max_inputs - min_inputs) + min_inputs
            
        # Convert volume array to tensor
        vol = tf.convert_to_tensor(element_vol, dtype=tf.float32) # (N_per_batch,)

        # Initialise total energy difference
        Energy_difference_epoch = tf.constant([[0.]])

        # Iterate over each Batch
        for batch_idx in range(0, len(inputs_unscaled), N_per_batch): # (0, 864x10, 864)

            print('Batch index: ',batch_idx)

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
            array_inputs=np.array(input_batch) # do we need this?
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
            Energy_difference_epoch += Energy_difference_batch
            print('Energy difference             - epoch   :',Energy_difference_epoch[0][0].numpy()); print(' ') # print newline

        return Energy_difference_epoch

    return custom_loss




#%%########################################################################################################
# Creating ML Model

def Create_Model(activ, Hidden_layers, node_num1, node_num2):
    
    model=tf.keras.Sequential()

    # Add the first hidden layer with the specified number of nodes and activation function
    # Input shape is set to match the shape of the input data
    model.add(tf.keras.layers.Dense(node_num1, activation=activ, input_shape=[3], name='hidden_layer_0'))
    
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


#Calling create model amd compiling model with optimizer

model = Create_Model(activ='linear', Hidden_layers=2, node_num1=20, node_num2=20)
model.summary()
model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001), 
          loss = custom_loss_with_params(external_work = external_work, 
                                         normalisation_params = normalisation_params,
                                         element_vol = element_vol,
                                         N_per_batch = N))




#%%########################################################################################################
# Train ML Model manually using train_on_batch

# Initialise vars
epoch_loss_list = [] 
batch_size = N # Train on entire specimen element set (864)
batches_per_epoch = scaled_inputs.shape[0] // batch_size # Number of time steps / data sets for that specimen
N_epochs = 50

# Iterate over each Epoch
for epoch in range(N_epochs):

    total_loss = 0.0 # sum of loss of all batches in this epoch
    print('Epoch number: ', epoch) ; print('')

    # Train all 10 timesteps at once (one batch of 8640 records)
    loss = model.train_on_batch(x=scaled_inputs, y=scaled_inputs, reset_metrics=True)

    # Track loss of all epochs for plotting
    epoch_loss_list.append(loss) 




#%%########################################################################################################
# Plot training loss history
    
N_epochs_range  = np.arange(1, N_epochs + 1)
plt.plot(N_epochs_range, epoch_loss_list, label='Training Loss per Epoch')
plt.ylabel('Training loss (energy difference) = sum of batch losses')
plt.xlabel('Epoch number')
plt.title('Training loss plot')




#%%########################################################################################################
# Getting output for last batch only (last time step)

'''    
    
L_finalT = tf.transpose(stacked_L,perm=[0,2,1])

############################################################################################################################        
# Calculate C
C_final = tf.matmul(stacked_L, L_finalT) # size: (N,3,3) 


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
'''
