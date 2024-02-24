import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(fig, epoch_loss_list, epoch_time_list):

    # Loss vs Epochs subplot
    fig1 = fig.add_subplot(2,1,1)
    N_epochs_range  = np.arange(1, len(epoch_loss_list) + 1)
    fig1.plot(N_epochs_range, epoch_loss_list, label='Training loss')
    fig1.grid()
    fig1.set_ylabel('Training loss')
    fig1.set_xlabel('Epoch number')
    fig1.set_title('Training loss history')
    fig1.set_ylim((1e1,4e4))

    # Loss vs Runtime subplot
    fig2 = fig.add_subplot(2,1,2)
    fig2.semilogy(epoch_time_list, epoch_loss_list)
    fig2.grid()
    fig2.set_ylabel('Training loss')
    fig2.set_xlabel('Runtime')
    fig2.set_ylim((1e1,4e4))
    plt.tight_layout()

    return