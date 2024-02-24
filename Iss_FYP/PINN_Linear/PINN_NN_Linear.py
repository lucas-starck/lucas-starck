#%%
# Import Modules
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt

# Import Custom functions
from import_data import import_data 
from custom_loss_with_params import custom_loss_with_params
from custom_train import custom_train
from plot_loss_curves import plot_loss_curves
from predict_stress import predict_stress
from calc_plot_PR import calc_plot_PR
from plot_stress_contours import plot_stress_contours
from create_model import create_model
from calc_plot_YM import calc_plot_YM



#%%
# Import Data

scaled_inputs, external_work, normalisation_params, element_vol, Pmax, x_coords, y_coords, stress_data, N = import_data()



#%%
# Create & Compile ML Model

model = create_model('linear', Hidden_layers=3, node_num1=20, node_num2=20)
model.summary()



#%%
#Calling create model amd compiling model with optimizer

epochs = 100

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), 
          loss = custom_loss_with_params(external_work = external_work, 
                                         normalisation_params = normalisation_params,
                                         element_vol = element_vol,
                                         N_per_batch = N,
                                         Pmax = Pmax))

training_loss, time_loss = custom_train(scaled_inputs , model, N_epochs = epochs)

predicted_stress, C = predict_stress(Pmax, model, scaled_inputs, normalisation_params)



#%%
## Plot training loss history

fig = plt.figure()
plot_loss_curves(fig, training_loss, time_loss)



#%%
## Plot stress contour plots

plot_stress_contours(x_coords, y_coords, predicted_stress, N) 
plot_stress_contours(x_coords, y_coords, stress_data, N)



#%%
## Plot Poisson Ratio and Youngs Modulus 

error_PR, poisson = calc_plot_PR(x_coords, y_coords, C, N, true_PR = 0.34)
error_YM = calc_plot_YM(x_coords, y_coords, C, N, poisson, true_YM = 1.65e5)



# %%
