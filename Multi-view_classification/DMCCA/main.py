try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle


import time
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import create_model
from load_data_l3 import load_noisy_mnist_data
import matplotlib.pyplot as plt

start = time.time()


def train_model(model, data_list, epoch_num, batch_size, feature_dim):

    # Unpacking the data
    # the data_list is arranged thus:
    # [[(train_x, train_y), (val_x, val_y), (test_x, test_y) ]_(1), {}_(2),...]
    #print("data list",len(data_list))

    train_x_list = [i[0][0] for i in data_list]
    val_x_list = [i[1][0] for i in data_list]
    print("train_x_list 0",len(train_x_list[0]))
    print("train_x_list 1",len(train_x_list[1]))
    print("train_x_list 2",len(train_x_list[2]))
    test_x_list = [data_list[0][2][0] for i in data_list]
    test_x_list2 = [data_list[1][2][0] for i in data_list]
    test_x_list3 = [data_list[2][2][0] for i in data_list]
    #print("train_x1_list",len(test_x_list2[0]))
    #print("train_x2_list",len(test_x_list3[0]))
    
    # for later
    test_y_list = [data_list[0][2][1] for i in data_list]
    test_y_list2 = [data_list[1][2][1] for i in data_list]
    test_y_list3 = [data_list[2][2][1] for i in data_list]

    # it is done to return the best model based on the validation loss
    #checkpointer = ModelCheckpoint(filepath="model_weights/weights_%d_dim.{epoch:02d}-{val_loss:.4f}.hdf5" % (feature_dim), 
    #                                    verbose=1, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(min_delta = 1e-4, patience = 5)

    # used dummy Y because labels are not used in the loss function
    history = model.fit(train_x_list, np.zeros(len(train_x_list[0])),
              batch_size=batch_size, epochs=epoch_num, shuffle=True,
              validation_data=(val_x_list, np.zeros(len(val_x_list[0]))),callbacks=[early_stopping])
    
    print("history",history.history.keys())
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    #model_names_ = glob.glob('weights*5')
    #model.load_weights(model_names_[-1])

    results = model.evaluate(test_x_list, np.zeros(len(test_x_list[0])), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results)
    
    results2 = model.evaluate(test_x_list2, np.zeros(len(test_x_list2[0])), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results2)
    
    results3 = model.evaluate(test_x_list3, np.zeros(len(test_x_list3[0])), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results3)



    results = model.evaluate(val_x_list, np.zeros(len(val_x_list[0])), batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)
    return model


def test_model(model, data_list, apply_mcca=False):
    """produce the new features by using the trained model
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
    """

    # the data_list is arranged thus:
    # [[(train_x, train_y), (val_x, val_y), (test_x, test_y) ]_(1), {}_(2),...]
    train_x_list = [i[0][0] for i in data_list]
    train_x_list1 = [data_list[0][0][0] for i in data_list]
    train_x_list2 = [data_list[1][0][0] for i in data_list]
    train_x_list3 = [data_list[2][0][0] for i in data_list]
    train_x_list4 = [data_list[3][0][0] for i in data_list]
    train_x_list5 = [data_list[4][0][0] for i in data_list]
    train_x_list6 = [data_list[4][0][0] for i in data_list]
 
    val_x_list = [i[1][0] for i in data_list]
    test_x_list = [data_list[0][2][0] for i in data_list]
    test_x_list1 = [data_list[0][2][0] for i in data_list]
    test_x_list2 = [data_list[1][2][0] for i in data_list]
    test_x_list3 = [data_list[2][2][0] for i in data_list]
    test_x_list4 = [data_list[3][2][0] for i in data_list]
    test_x_list5 = [data_list[4][2][0] for i in data_list]
    test_x_list6 = [data_list[5][2][0] for i in data_list]

    
    # for later
    train_y = [i[0][1] for i in data_list][0] # since all three modalities have same labels
    val_y = [i[1][1] for i in data_list][0]
    test_y = [data_list[0][2][1] for i in data_list]
    test_y2 = [data_list[1][2][1] for i in data_list]
    test_y3 = [data_list[2][2][1] for i in data_list]
    # producing the new features
    train_embeddings = model.predict(train_x_list)

    save_path = "/home/akansha/RESEARCH/1_current/My_DMCCA/2018/features_l3/"
    
    ''' computing embeddings for train data'''
    train_embeddings1 = model.predict(train_x_list1)
    np.save(save_path+"emb_train_v1",train_embeddings1)
    train_embeddings2 = model.predict(train_x_list2)
    np.save(save_path+"emb_train_v2",train_embeddings2)
    train_embeddings3 = model.predict(train_x_list3)
    np.save(save_path+"emb_train_v3",train_embeddings3)
    train_embeddings4 = model.predict(train_x_list4)
    np.save(save_path+"emb_train_v4",train_embeddings4)
    train_embeddings5 = model.predict(train_x_list5)
    np.save(save_path+"emb_train_v5",train_embeddings5)
    train_embeddings6 = model.predict(train_x_list6)
    np.save(save_path+"emb_train_v6",train_embeddings6)
    print("train embeddings",train_embeddings.shape)
    
    val_embeddings = model.predict(val_x_list)
    
    ''' computing embeddings for test data'''
    test_embeddings1 = model.predict(test_x_list1)
    np.save(save_path+"emb_test_v1",test_embeddings1)
    test_embeddings2 = model.predict(test_x_list2)
    np.save(save_path+"emb_test_v2",test_embeddings2)
    test_embeddings3 = model.predict(test_x_list3)
    np.save(save_path+"emb_test_v3",test_embeddings3)
    test_embeddings4 = model.predict(test_x_list4)
    np.save(save_path+"emb_test_v4",test_embeddings4)
    test_embeddings5 = model.predict(test_x_list5)
    np.save(save_path+"emb_test_v5",test_embeddings5)
    test_embeddings6 = model.predict(test_x_list6)
    np.save(save_path+"emb_test_v6",test_embeddings6)


    return [(train_embeddings, train_y),(train_embeddings1, train_y),(train_embeddings2, train_y),
            (train_embeddings3, train_y),(train_embeddings4, train_y),(train_embeddings5, train_y),
            (train_embeddings6, train_y), (val_embeddings,val_y), (test_embeddings1, test_y), 
            (test_embeddings2, test_y2), (test_embeddings3, test_y3),(test_embeddings4, test_y), 
            (test_embeddings5, test_y2), (test_embeddings6, test_y3)]


if __name__ == '__main__':
############
# Parameters Section

# the path to save the final learned features
    save_to = './mcca_noisy_mnist_features.gz'

# number of modalities/datasets = n_mod
    n_mod = 6

# size of the input for view 1 and view 2
    input_shapes = [512, 512, 512,512,512,512]

# the size of the new space learned by the model (number of the new features)
    outdim_size = 9 # has to be same for all modalities - (TODO) change later

# layer size list to create a simple FCN
    layer_size_list = [[256,128,64,32, outdim_size]] * n_mod

# the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 100
    batch_size = 128

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False

# end of parameters section
############

# the load_data function loads noisy mnist data as a list of train,val,test triples
    data_list = load_noisy_mnist_data()
    print("data list",len(data_list))
    #print("data list view 0",len(data_list[0]))
    #print("data list view 1",len(data_list[1]))
    #print("data list view 2",len(data_list[2]))
    print("view 0 train",len(data_list[2][0][0]))
    print("view 0 val",len(data_list[2][1][0]))
    print("view 0 test",len(data_list[2][2][0]))
    #print("view 0 train",data_list[0][0][1])
    

# Building, training, and producing the new features by DCCA
    model = create_model(layer_size_list, input_shapes, 
                            act_='linear', learning_rate=learning_rate, n_modalities=n_mod,gamma=0.9, reg_par=1e-5)
    model.summary()
    model = train_model(model, data_list, epoch_num, batch_size, outdim_size)
    model.save('saved_model_sigmoid_at_last_layer.h5')
    
    data_embeddings = test_model(model, data_list)
# just test on the test set for now to assess the viability!
    #pred_on_test = run_svc_pipeline_doubleCV(data_embeddings[2][0], data_embeddings[2][-1])
    #print(pred_on_test[-1])
    #print("data_embeddings[0]",data_embeddings[0])

    #np.savez('saved_embeddings_sigmoid_at_last_layer_relu_model', train=data_embeddings[0], val=data_embeddings[1], test=data_embeddings[2])
end = time.time()
print("time taken",end-start)