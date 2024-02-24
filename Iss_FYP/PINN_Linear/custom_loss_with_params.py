import tensorflow as tf
import numpy as np

#Designing custom loss function
#Wrapper function
def custom_loss_with_params(external_work, normalisation_params, element_vol, N_per_batch, Pmax):

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
            #print('Batch number: ',batch_number)

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
            #print('Internal work (strain energy) - batch',batch_number,':', strain_energy_batch[0][0].numpy())
            
            #print('External work (F x d)         - batch',batch_number,':', external_work[batch_number]) 
            
                
            # Calculate Loss for batch: 
            #   Difference between internal (strain energy) and external work (F*d)
            Energy_difference_batch = tf.abs((external_work[batch_number]/Pmax)-((strain_energy_batch)/Pmax))
            scaled_internal_energy = strain_energy_batch/Pmax
            scaled_external_work = external_work[batch_number]/Pmax
            #print('Scaled Internal work          - batch',batch_number,':', scaled_internal_energy[0][0].numpy())
            #print('Scaled External work (F x d)  - batch',batch_number,':', scaled_external_work.numpy()) 
            # Add batch energy difference to total epoch energy difference
            # Accumulate Energy_difference_batch to the array
            Energy_difference_epoch.append(Energy_difference_batch)
            Energy_difference_epoch_total = tf.concat([Energy_difference_epoch], axis=0)
            
            
        #print(Energy_difference_epoch_total.shape)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(Energy_difference_epoch_total)))
        print('RMSE:', np.array(rmse)); print(' ') # print newline

        return rmse     

    return custom_loss