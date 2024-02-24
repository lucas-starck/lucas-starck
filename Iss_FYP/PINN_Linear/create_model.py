import tensorflow as tf


def create_model(activ, Hidden_layers, node_num1, node_num2):
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