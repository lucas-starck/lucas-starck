#%%
#Imports
import numpy as np
import tensorflow as tf
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
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
element_volume = df.iloc[:, 8].values.astype('float32') # Volume of each element for volume integral size: (9504,1)

standardisation_params = tf.Variable(tf.zeros((2,4)), trainable=False)
         
#Combining the features into one array
inputs = tf.stack([Exx_data, Eyy_data, Exy_data, element_volume], axis=1) #size: (9504, 3)

#calculating external work
external_work =0.5* (P_data * Ux_data) # gives a vector of external work values at each time step/batch size: (11, 1)

print(external_work)

#Designing custom loss function

#%%
def custom_loss_with_params(external_work, standardisation_params):
    def custom_loss(inputs, L):   

        #unscaling the inputs
        
        max_inputs, min_inputs = tf.split(standardisation_params, 2)

        reversed_inputs = inputs * (max_inputs - min_inputs) + min_inputs

        #undoing the inputs normalisation

        '''
        bn_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                bn_layer = layer
                break

        if bn_layer:
            # Get the learned parameters (gamma and beta) and running averages
            gamma, beta = bn_layer.get_weights()[:2]
            mean, variance = bn_layer.moving_mean, bn_layer.moving_variance

            # Unscale the inputs
            epsilon = bn_layer.epsilon
            unscaled_inputs = gamma * (inputs - beta) / tf.sqrt(variance + epsilon) + mean
        
        else:

            print("BatchNormalization layer not found in the model.")
        '''
        array_inputs=np.array(reversed_inputs)
        
        Exx = array_inputs[:,0]
        Eyy = array_inputs[:,1]
        Exy = array_inputs[:,2]
        vol = array_inputs[:,3] 
        
        vol_tf=tf.convert_to_tensor(vol, dtype=tf.float32)
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
       

        max_values = tf.reduce_max(L, axis=-1, keepdims=True)

         
        # Before passing the training data, L is size (none,9). But during training, L is size (N,9). 
        '''
        if L.shape[0]!=L.shape[0]:
            L_matrix = tf.reshape(L,(1,3,3)) #output of Neural network such that C=LL^T - size should be a (N,3,3)
        else: 
            L_matrix = tf.reshape(L,(N,3,3)) #output of Neural network such that C=LL^T - size should be a (N,3,3)
        #print(L_matrix.shape)
        '''
    
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
        
        difference = tf.abs(1-(strain_energy/external_work[batch_no]))# calculates the squared difference between virtual work at the current batch and the strain energy - this may not be right

       
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
    model.add(tf.keras.layers.Dense(node_num1, activation=activ, input_shape=[4], name='hidden_layer_0'))

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

model = Create_Model('linear',3,40,40)
model.summary()
model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001), 
          loss = custom_loss_with_params(external_work=external_work, standardisation_params=standardisation_params)) #how do i make a custom loss function here?



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
batch_size = N
epochs = 50

# Initialize max and min values
#max_values = np.zeros((9,))
#min_values = np.zeros((9,))

for epoch in range(epochs):

    total_loss = 0.0
    batches_per_epoch = inputs.shape[0] // batch_size
    print('epoch number', epoch)
    custom_loss_calls = tf.Variable(0, trainable=False, dtype=tf.int32) # intialising counter for number of times loss function is called
    
    for i in range(0, len(inputs), batch_size):
        batch_no = custom_loss_calls.numpy()
        print('batch number',batch_no)
    
        x_batch = inputs[i:i+batch_size]
        
        #performing feature scaling per batch instead of on whole dataset
        max_x_batch = tf.reduce_max(x_batch, axis=0)
        min_x_batch = tf.reduce_min(x_batch, axis=0)
        
        standardisation_params.assign_add([max_x_batch, min_x_batch])
        #print('shape of stand_params:', standardisation_params.shape)
        scaled_x_batch = (x_batch-min_x_batch)/(max_x_batch-min_x_batch)
        
        y_batch = scaled_x_batch

        custom_loss_calls.assign_add(1) # adds one to the counter for how many times this function is entered/ which batch we are on
        # Train on the batch
        loss = model.train_on_batch(scaled_x_batch, y_batch, reset_metrics=True)
        L_batch = model.predict_on_batch(scaled_x_batch)
        #max_values = np.maximum(max_values, np.max(L_batch, axis=0))
        #min_values = np.minimum(min_values, np.min(L_batch, axis=0))

        total_loss += loss

    average_loss = tf.sqrt(tf.square(total_loss)/ batches_per_epoch)
    losses.append(average_loss)


#getting output for last batch only (last time step)
L_finalT = tf.transpose(L_batch,perm=[0,2,1])

############################################################################################################################        
# Calculate C
C_final = tf.matmul(L_batch, L_finalT) # size: (N,3,3) 

Exx = x_batch[:,0]
Eyy = x_batch[:,1]
Exy = x_batch[:,2]

epsilon = tf.stack([Exx, Eyy, Exy], axis=1)
epsilon = tf.reshape(epsilon , (epsilon.shape[0],3,1)) #sise:


stress = tf.matmul(C_final, epsilon)
print('stress shape:', stress.shape)

reshaped_tensor = tf.reshape(stress, (stress.shape[0], -1))

numpy_array = reshaped_tensor.numpy()

# Specify the file path for the CSV file
csv_file_path = 'output_stress.csv'

# Use NumPy's savetxt function to save the array to a CSV file
np.savetxt(csv_file_path, numpy_array, delimiter=',')

print('stress at final batch:', stress)
print('stress type:', type(stress))

print('final L:', L_batch)
print('final loss:', average_loss)

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