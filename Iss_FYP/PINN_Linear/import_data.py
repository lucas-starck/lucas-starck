########################################################################################################
from glob import glob
import os
import numpy as np
import tensorflow as tf

# Extracting FEA stress output
def import_data():
    #importing coordinates and number of elements
    root_file_path = '/Users/lucasastarck/Documents/Data Science/Coding/lucas-starck/Iss_FYP/PINN_Linear/FEA_data/'
    inp_file_path = root_file_path+'Plate_hole_1.inp'
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


    no_batches = 14
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

    Sxx_data = final_data_stress[:, 0]
    Syy_data = final_data_stress[:, 1]
    Sxy_data = final_data_stress[:, 2]

    stress_data = tf.stack([Sxx_data, Syy_data, Sxy_data], axis=1)


    #outputting volume and displacement/force data from fea

    filename_vol = 'volume.txt'
    file_path_vol = glob(os.path.join(root_file_path, filename_vol))
    element_vol = np.loadtxt(file_path_vol[0], dtype='float32', usecols=(1), skiprows=22, max_rows=N)


    filename_ext_work = 'force_displacement_data.txt'
    file_path_ext_work = glob(os.path.join(root_file_path, filename_ext_work))
    data_ext_work = np.loadtxt(file_path_ext_work[0], dtype='float32', usecols=(0, 2), skiprows=4, max_rows=N)

    Ux_data = data_ext_work[1:,0] 
    P_data = data_ext_work[1:,1]
    Pmax = tf.reduce_max(P_data) 


    #P_data*=1e3 # trying to put the unit N into mm

    # Extracting scaling values to scale between (0,1) and storing them
    #   Note: we are scaling by the max/min value of all 10 batches, and not of each batch
    inputs = tf.stack([Exx_data, Eyy_data, Exy_data], axis=1) #size: (9504, 3)
    max_inputs = tf.reduce_max(inputs, axis=0)
    min_inputs = tf.reduce_min(inputs, axis=0)
    normalisation_params = tf.Variable(tf.zeros((2,3)), trainable=False)
    normalisation_params.assign_add([max_inputs, min_inputs])

    scaled_inputs = (inputs-min_inputs)/(max_inputs-min_inputs)

    # Calculate the index from where the last 3 batches start
    #validation_start_index = len(scaled_inputs) - 3 * N #3 is the number of batches i want to use for validation

    # Split the data into training and validation sets
    #train_inputs = scaled_inputs[:validation_start_index, :]
    #val_inputs = scaled_inputs[validation_start_index:, :]

    #print("Training data shape:", train_inputs.shape)
    #print("Validation data shape:", val_inputs.shape)
    #calculating external work
    external_work =0.5* (P_data * Ux_data) # gives a vector of external work values at each time step/batch size: (11, 1)
    #print('External work: '); print(external_work); print('')

    return scaled_inputs, external_work, normalisation_params, element_vol, Pmax, x_coords, y_coords, stress_data, N