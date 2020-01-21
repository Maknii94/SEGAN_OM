from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Input, Conv1D
from keras.layers import BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(filters=31,kernel_size=1,strides=2)(X_input)       # CONV1D
    X = Conv1D(filters=31,kernel_size=16,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=32,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=32,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=64,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=64,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=128,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=128,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=256,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=256,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=512,strides=2)(X)
    X = Conv1D(filters=31,kernel_size=1024,strides=2)(X)
    X = BatchNormalization()(X)                                   # Batch normalization
    X = Activation("relu")(X)                                     # ReLu activation
    X = Dropout(0.8)(X)                                           # dropout (use 0.8)

    model = Model(inputs = X_input, outputs = X)
    
    return model