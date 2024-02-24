import tensorflow as tf
import numpy as np
import timeit


# Train ML Model manually using train_on_batch

def custom_train(scaled_inputs, model, N_epochs=500, ):
    epoch_loss_list = []
    #validation_loss_list=[]
    epoch_time_list = []
    epoch_time_cumsum = 0.

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

        # Evaluate on validation data
        #validation_loss = model.test_on_batch(x=val_inputs, y=val_inputs)
        #validation_loss_list.append(validation_loss) 
        


        # Break training if runtime limit is exceeded
        if epoch_time_cumsum > 5*60.:
            break

    return epoch_loss_list, epoch_time_list

