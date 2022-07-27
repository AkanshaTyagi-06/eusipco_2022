from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
from objectives import cca_loss
from objectives_mcca import mcca_loss

def create_model(layer_sizes_list, input_size_list, act_='linear', 
                            learning_rate=1e-3, n_modalities=3, gamma=0.2, reg_par=1e-1):
    """
    Input:
    ..
    Output:
    ..

    builds the whole model form a list of list of layer sizes!
    !!## note this is not the Sequential style model!
    """    
    print("gamma in create model",gamma)
    input_layers = [Input((size_i, )) for size_i in input_size_list]

    fc_output_layer_list = []

    for l_i, layer_sizes_ in enumerate(layer_sizes_list):
        # pre-create the dense(fc) layers you need
        ## USING ONLY LINEAR ACTIVATIONS FOR NOW!!
        fc_layers_ = [Dense(i,activation=act_, kernel_regularizer=l2(reg_par)) for i in layer_sizes_[:-1]]
        # no matter the layer activation, the last layer needs a sigmoid activation!
        fc_layers_.append(Dense(layer_sizes_[-1], activation=act_, kernel_regularizer=l2(reg_par)))

        D = fc_layers_[0](input_layers[l_i])
        # do this in a non-sequential style Keras model
        for d_i, d in enumerate(fc_layers_[1:]): D = d(D) 
        fc_output_layer_list.append(D)

    output = concatenate(fc_output_layer_list)
    model = Model(input_layers, [output])

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=mcca_loss(n_modalities, gamma), optimizer=model_optimizer)

    return model



