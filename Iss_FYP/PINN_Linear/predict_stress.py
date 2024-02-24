import tensorflow as tf
import numpy as np


def predict_stress(Pmax, model, scaled_inputs, normalisation_params):
    #predciting L at each epoch - overwrites but that's okay because i only want last epoch
    L_epoch = model.predict_on_batch(scaled_inputs)

    #Calculates L and LT as tensors
    L_epoch_tensor = tf.convert_to_tensor(L_epoch, dtype=tf.float32)# size: (8640,3,3) 
    L_finalT = tf.transpose(L_epoch_tensor,perm=[0,2,1])# size: (8640,3,3) 
        
    # Calculate C prediction (at every example)
    C_final = tf.matmul(L_epoch_tensor, L_finalT) # size: (8640,3,3) 

    #print('C final:', C_final)
    #print('C final shape:', C_final.shape)

    max_inputs, min_inputs = tf.split(normalisation_params, 2)
    inputs_unscaled = scaled_inputs * (max_inputs - min_inputs) + min_inputs

    array_inputs=np.array(inputs_unscaled) # do I need this?
    Exx_data = array_inputs[:,0]
    Eyy_data = array_inputs[:,1]
    Exy_data = array_inputs[:,2]

    # Calculate epsilon and its transpose (for the batch)
    epsilon = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1) # size: (N, 3)
    epsilon = tf.reshape(epsilon , (epsilon.shape[0],3,1))

    #Calculate the stress stress=Cxepsilon
    stress_predicted = tf.matmul(C_final, epsilon) 

    #reshaping stress for output
    stress_predicted = tf.reshape(stress_predicted, (stress_predicted.shape[0], -1))
    numpy_array = stress_predicted.numpy()

    # Specify the file path for the CSV file
    csv_file_path = 'output_stress.csv'

    # Use NumPy's savetxt function to save the stress array to a CSV file
    np.savetxt(csv_file_path, numpy_array, delimiter=',')

    return stress_predicted, C_final