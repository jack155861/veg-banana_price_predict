#!pip install focal_loss
from tensorflow.keras.layers import \
    BatchNormalization, Dropout, LayerNormalization, \
    Input, Conv1D, Activation, concatenate, Dense, Flatten, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, \
    LSTM, TimeDistributed, \
    Conv2D, MaxPool2D, GlobalMaxPool2D, MaxPooling2D, \
    PReLU, LeakyReLU, Bidirectional, ConvLSTM2D, RepeatVector
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm as maxnorm
from tensorflow.keras.utils import plot_model, get_custom_objects
from tensorflow.keras.optimizers import SGD, Nadam, Adam
from tensorflow.keras.backend import sigmoid
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
from tensorflow.keras.activations import gelu
import matplotlib.pyplot as plt
#from focal_loss import SparseCategoricalFocalLoss
from itertools import product
from tensorflow.keras.losses import sparse_categorical_crossentropy
import argparse
from multiprocessing.dummy import Pool as ThreadPool

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--data_input', type=str)  
    #parser.add_argument('--activation_conv', type=str)
    #parser.add_argument('--activation_dense', type=str)
    opt = parser.parse_args()
    return opt

def parameter(opt, activation_conv, activation_dense):
    original_input = {
        'model_type':opt.model_type,
        'data_input':opt.data_input,
        'activation_conv':activation_conv ,#opt.activation_conv,
        'activation_dense':activation_dense #opt.activation_dense
    }
    
    parameter_input = []
    for optimizer in ["nadam", "sgd", "adam"]:
        for l2_rate in [0, 0.0001, 0.001, 0.01, 0.1]:
            tmp = {"optimizer":optimizer,"l2_rate":l2_rate}
            tmp.update(original_input)
            parameter_input.append(tmp)
    return parameter_input

def class_weight_cal(col, data_):
    class_weights = compute_class_weight(class_weight = 'balanced', 
                                         classes = np.array(data_[col]), 
                                         y = data_[col].values)
    class_weights = dict(zip(np.array(data_[col]).astype(int), class_weights))
    
    return {"day"+str(int(col[-1])-1):class_weights}

def swish(x, beta = 1):
    # https://www.geeksforgeeks.org/ml-swish-function-by-google-in-keras/
    return (x * sigmoid(beta * x))
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.25)
def relu_fix(x):
    return tf.keras.activations.relu(x, max_value = 1)

get_custom_objects().update({'swish': swish})
get_custom_objects().update({'leaky_relu': leaky_relu})
get_custom_objects().update({'relu_fix': relu_fix})

def optimizer_f(optimizer):
    if optimizer == "sgd":
        return SGD(learning_rate=0.001, momentum=0.9, decay=0, nesterov=True)
    if optimizer == "nadam":
        return Nadam(lr=0.001)
    if optimizer == 'adam':
        return Adam(learning_rate=0.001)

def image_convert(a):
    new = np.zeros((30, 30))
    for i in range(30):
        if int(a[i]*100)==0:
            d = 29
        elif int(a[i]*100)==100:
            d = 0
        else:
            d = 30-int(30-(1-a[i])*30)
            if d==30:
                d = 29
        new[d, i] = 1
    return new

def callbacks_mutilabel(file_name, factor = 0.5):
    try:
        os.mkdir(file_name)
    except:
        pass
    
    # optimizers 
    lrate = ReduceLROnPlateau(monitor='val_accuracy', factor=factor, patience=25, mode='min', min_delta=0.00001, 
                              cooldown=0, threshold_mode='rel', min_lr=1e-8, verbose = 2)    
    earlyst = EarlyStopping(monitor='val_accuracy', patience = 200, mode='min', min_delta=0.00001, verbose=2)    

    # save weight
    #ck_callback_loss = ModelCheckpoint(file_name + '/weights_loss.hdf5',#_{epoch:04d}
    #                                   monitor="val_loss", mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    #ck_callback_loss._supports_tf_logs = False   
    ck_callback_acc = ModelCheckpoint(file_name + '/weights_accuracy.hdf5',#_{epoch:04d}
                                      monitor="val_accuracy", mode='max', verbose=0, save_best_only=True, save_weights_only=False)
    ck_callback_acc._supports_tf_logs = False  
    
    ## https://stackoverflow.com/questions/58391418/how-to-use-custom-metric-from-a-callback-with-earlystopping-or-modelcheckpoint
    csv_callback = CSVLogger(file_name + '/log.csv', append=True, separator=',')
    
    return [CombinedMetric(),
            ck_callback_loss, 
            ck_callback_acc, 
            csv_callback, 
            lrate, 
            earlyst]

def callbacks_regression(file_name, factor = 0.5):
    try:
        os.mkdir(file_name)
    except:
        pass
    
    # optimizers 
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=25, mode='min', min_delta=0.00001, 
                              cooldown=0, threshold_mode='rel', min_lr=1e-8, verbose = 2)    
    earlyst = EarlyStopping(monitor='val_loss', patience = 200, mode='min', min_delta=0.00001, verbose=2)    

    # save weight
    ck_callback_loss = ModelCheckpoint(file_name + '/weights_loss.hdf5',#_{epoch:04d}
                                       monitor="val_loss", mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    ck_callback_loss._supports_tf_logs = False   
    ck_callback_acc = ModelCheckpoint(file_name + '/weights_mse.hdf5',#_{epoch:04d}
                                      monitor="val_mse", mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    ck_callback_acc._supports_tf_logs = False  
    #ck_callback_acc = ModelCheckpoint(file_name + '/weights_accuracy.hdf5',#_{epoch:04d}
    #                                  monitor="val_accuracy", mode='max', verbose=0, save_best_only=True, save_weights_only=False)
    #ck_callback_acc._supports_tf_logs = False  
    
    
    ## https://stackoverflow.com/questions/58391418/how-to-use-custom-metric-from-a-callback-with-earlystopping-or-modelcheckpoint
    csv_callback = CSVLogger(file_name + '/log.csv', append=True, separator=',')
    
    return [ck_callback_loss, 
            ck_callback_acc, 
            csv_callback, 
            lrate, 
            earlyst]

def Normalize_f(input_, method, l2_rate_, name, lstm = False):
    if method=="Batch":
        if lstm:
            em_layer = TimeDistributed(BatchNormalization(beta_regularizer=l2(l2_rate_), gamma_regularizer=l2(l2_rate_), name = name))(input_)
        else:
            em_layer = BatchNormalization(beta_regularizer=l2(l2_rate_), gamma_regularizer=l2(l2_rate_), name = name)(input_)
    if method=="Layer":
        if lstm:
            em_layer = TimeDistributed(LayerNormalization(beta_regularizer=l2(l2_rate_), gamma_regularizer=l2(l2_rate_), name = name))(input_)
        else:
            em_layer = LayerNormalization(beta_regularizer=l2(l2_rate_), gamma_regularizer=l2(l2_rate_), name = name)(input_)
    if method=="dropout":
        if lstm:
            em_layer = TimeDistributed(Dropout(l2_rate_, name = name))(input_)
        else:
            em_layer = Dropout(l2_rate_, name = name)(input_)
    return em_layer

def non_time_series_model(train_X_num_, activation_ = 'relu', unit = 1024, l2_rate_ = 0.01):
    model2_in = Input(shape=train_X_num_.shape[1:])
    model2_out = Dense(unit, activation=activation_, name = 'non_times_dense_1', kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate_))(model2_in)
    model2_out = Dense(int(unit/4), activation=activation_, name = 'non_times_dense_2', kernel_initializer='he_normal')(model2_out)
    return model2_in, model2_out

def output_model_2(concat, output_activation, l2_rate, activation_ = 'relu'):
    out = Dense(1024, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(concat)
    out = Dropout(0.5)(out)
    out = Dense(256, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(64, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out1 = Dense(3, activation=output_activation, name = "day1", kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out2 = Dense(3, activation=output_activation, name = "day2", kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out3 = Dense(3, activation=output_activation, name = "day3", kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out4 = Dense(3, activation=output_activation, name = "day4", kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out5 = Dense(3, activation=output_activation, name = "day5", kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    return out1, out2, out3, out4, out5

def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history[0])
    plt.plot(train_history[1])
    plt.title("Train History")
    plt.xlabel("Epoch")
    #plt.ylabel("Accuracy")
    plt.ylabel("MSE")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history[2])
    plt.plot(train_history[3])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.close(fig)
    return fig

def calculating_class_weights(y_true):
    # https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950
    #return dict(zip(np.unique(y_true).astype(int),
    #                compute_class_weight(class_weight = 'balanced', classes = np.unique(y_true), y = y_true)))
    return compute_class_weight(class_weight = 'balanced', classes = np.unique(y_true), y = y_true)

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
def output_model_3(concat, output_activation, l2_rate, activation_ = 'relu', activation_1 = 'relu'):
    out = Dense(1024, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(concat)
    out = Dropout(0.5)(out)
    out = Dense(256, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(64, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = RepeatVector(5)(out)
    out = LSTM(64, activation=activation_1, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    #out = TimeDistributed(Dense(3, activation=output_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate)))(out)
    #regression
    out = TimeDistributed(Dense(1, activation=output_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate)))(out)
    return out
class CombinedMetric(Callback):
    def __init__(self):
        super(CombinedMetric, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        try:
            logs['accuracy'] = (logs["day1_accuracy"] + logs["day2_accuracy"] + logs["day3_accuracy"] + logs["day4_accuracy"] + logs["day5_accuracy"]) / 5
            logs['val_accuracy'] = (logs["val_day1_accuracy"] + logs["val_day2_accuracy"] + logs["val_day3_accuracy"] + logs["val_day4_accuracy"] + logs["val_day5_accuracy"]) / 5
            #logs['categorical_accuracy'] = (logs["day1_categorical_accuracy"] + logs["day2_categorical_accuracy"] + logs["day3_categorical_accuracy"] + logs["day4_categorical_accuracy"] + logs["day5_categorical_accuracy"]) / 5
            #logs['val_categorical_accuracy'] = (logs["val_day1_categorical_accuracy"] + logs["val_day2_categorical_accuracy"] + logs["val_day3_categorical_accuracy"] + logs["val_day4_categorical_accuracy"] + logs["val_day5_categorical_accuracy"]) / 5
        except:
            pass
        #print(" ??? loss: %f ??? val_loss: %f ??? accuracy: %f ??? val_accuracy: %f ??? ca_accuracy: %f ??? val_ca_accuracy: %f" % (logs['loss'], logs['val_loss'], logs['accuracy'], logs['val_accuracy'], logs['categorical_accuracy'], logs['val_categorical_accuracy']))
        print(" ??? loss: %f ??? val_loss: %f ??? accuracy: %f ??? val_accuracy: %f" % (logs['loss'], logs['val_loss'], logs['accuracy'], logs['val_accuracy']))

banana_X_train = pd.read_pickle("banana_X_train.pkl")
banana_X_val = pd.read_pickle("banana_X_val.pkl")
banana_Y_train = pd.read_pickle("banana_Y_train.pkl")
banana_Y_val = pd.read_pickle("banana_Y_val.pkl")
cols_name = [[y+'-'+str(x).rjust(2,"0") for x in range(1,31)] for y in ['RSV','???????????????','???????????????','???????????????','?????????','?????????','??????']]
cols_name_2 = [x for x in banana_X_train.columns.tolist() if '-' not in x]

train_X_num = banana_X_train[cols_name_2[1:]].values
val_X_num = banana_X_val[cols_name_2[1:]].values

# shape (samples, 7 ???????????????, 7 ???????????????)
train_X_lstm_1 = []
for i in range(len(cols_name)):
    tmp = banana_X_train[cols_name[i]].iloc[:,-7:]
    train_X_lstm_1.append(tmp)
train_X_lstm_1 = np.stack(train_X_lstm_1,axis=2)
val_X_lstm_1 = []
for i in range(len(cols_name)):
    tmp = banana_X_val[cols_name[i]].iloc[:,-7:]
    val_X_lstm_1.append(tmp)
val_X_lstm_1 = np.stack(val_X_lstm_1,axis=2)

# shape (samples, 7 ???????????????, 7 ????????????????????????????????????, 7 ???????????????)
train_X_lstm_2 = []
for i in range(len(cols_name)):
    tmp_all = []
    tmp = banana_X_train[cols_name[i]].iloc[:,-13:]
    for j in range(7):
        tmp_1 = tmp.iloc[:,j:j+7]
        tmp_all.append(tmp_1)
    tmp_all = np.stack(tmp_all,axis=2)
    train_X_lstm_2.append(tmp_all)
train_X_lstm_2 = np.stack(train_X_lstm_2,axis=3)

val_X_lstm_2 = []
for i in range(len(cols_name)):
    tmp_all = []
    tmp = banana_X_val[cols_name[i]].iloc[:,-13:]
    for j in range(7):
        tmp_1 = tmp.iloc[:,j:j+7]
        tmp_all.append(tmp_1)
    tmp_all = np.stack(tmp_all,axis=2)
    val_X_lstm_2.append(tmp_all)
val_X_lstm_2 = np.stack(val_X_lstm_2,axis=3)

# shape (samples, 7 ???????????????, 7 ???????????????????????????????????????, 1 ?????????????????????????????????, 7 ???????????????)
train_X_lstm_3 = np.expand_dims(train_X_lstm_2, axis=3)
val_X_lstm_3 = np.expand_dims(val_X_lstm_2, axis=3)

# shape (samples, 30 30????????????, 30 ??????????????????????????? 0-30,  7 ???????????????)
#plt.imshow(image_convert(banana_X_train[cols_name[0]].values[0])) 
#plt.show()
train_X_lstm_4 = np.stack([np.stack([image_convert(x) for x in banana_X_train[cols_name[y]].values]) for y in range(len(cols_name))], axis=3)
val_X_lstm_4 = np.stack([np.stack([image_convert(x) for x in banana_X_val[cols_name[y]].values]) for y in range(len(cols_name))], axis=3)

# shape (samples, 30 30????????????, 30 ?????????????????????????????????????????????, 30 ???????????????????????? 0-30,  7 ???????????????)
train_X_lstm_5 = []
for num in range(len(banana_X_train)):
    tmp_all_new=[]
    for feature_ in range(len(cols_name)):
        tmp = image_convert(banana_X_train[cols_name[feature_]].values[num])
        tmp_new = []
        for i in range(30):
            new = np.zeros((30, 30))
            new[:,:i+1] = tmp[:,:i+1]
            tmp_new.append(new)
        tmp_new = np.stack(tmp_new, axis=0)
        tmp_all_new.append(tmp_new)
    tmp_all_new = np.stack(tmp_all_new, axis=3)
    train_X_lstm_5.append(tmp_all_new)
train_X_lstm_5 = np.stack(train_X_lstm_5, axis=0)

val_X_lstm_5 = []
for num in range(len(banana_X_val)):
    tmp_all_new=[]
    for feature_ in range(len(cols_name)):
        tmp = image_convert(banana_X_val[cols_name[feature_]].values[num])
        tmp_new = []
        for i in range(30):
            new = np.zeros((30, 30))
            new[:,:i+1] = tmp[:,:i+1]
            tmp_new.append(new)
        tmp_new = np.stack(tmp_new, axis=0)
        tmp_all_new.append(tmp_new)
    tmp_all_new = np.stack(tmp_all_new, axis=3)
    val_X_lstm_5.append(tmp_all_new)
val_X_lstm_5 = np.stack(val_X_lstm_5, axis=0)

def lstm_model(train_X_lstm, type_, 
               Normalize_method = "Batch", Normalize_rate = 0.01,
               activation_lstm = "relu", activation_dense = "relu", unit_lstm = 100, unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    
    if type_=="Vanilla": #???????????????Vanilla LSTM
        model1_out = LSTM(unit_lstm, activation=activation_lstm, name = 'Vanilla', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate))(model1_in) 
    if type_=="Stacked": #???????????????Stacked LSTM
        model1_out = LSTM(unit_lstm, activation=activation_lstm, return_sequences=True, name = 'Stacked_1', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate))(model1_in) 
        model1_out = LSTM(unit_lstm, activation=activation_lstm, name = 'Stacked_2', kernel_initializer='he_normal')(model1_out) 
    if type_=="Bidirectional": #???????????????Bidirectional LSTM
        model1_out = Bidirectional(LSTM(unit_lstm, activation=activation_lstm, name = 'Bidirect', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate)))(model1_in) 
    
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method)
    model1_out = Dense(unit_dense, activation=activation_dense, name = 'times_dense_1', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate))(model1_out)
    model1_out = Dense(int(unit_dense/4), activation=activation_dense, name = 'times_dense_2', kernel_initializer='he_normal')(model1_out)
    return model1_in, model1_out

def conv1d_model(train_X_lstm, 
                 Normalize_method = "Batch", Normalize_rate = 0.01,
                 activation_conv = "relu", activation_dense = "relu", 
                 filters_conv = 64, kernel_conv = 2, pool_size=2,
                 unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    model1_out = Conv1D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_1')(model1_in)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_01")
    model1_out = Activation(activation_conv, name = 'times_act_1')(model1_out)
    model1_out = Conv1D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_2')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_02")
    model1_out = Activation(activation_conv, name = 'times_act_2')(model1_out)
    model1_out = MaxPooling1D(pool_size=pool_size, name = 'times_maxp_1')(model1_out)

    model1_out = Conv1D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_3')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_03")
    model1_out = Activation(activation_conv, name = 'times_act_3')(model1_out)
    model1_out = Conv1D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_4')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_04")
    model1_out = Activation(activation_conv, name = 'times_act_4')(model1_out)
    model1_out = MaxPooling1D(pool_size=pool_size, name = 'times_maxp_2')(model1_out)

    model1_out = Flatten()(model1_out)
    model1_out = Dense(unit_dense, activation=activation_dense, name = "times_dense_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(int(unit_dense/4), activation=activation_dense, name = "times_dense_2", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    return model1_in, model1_out

def conv1d_lstm_model(train_X_lstm, 
                      Normalize_method = "Batch", Normalize_rate = 0.01,
                      activation_conv = "relu", activation_dense = "relu", filters_conv = 64, kernel_conv = 2, pool_size=2,
                      unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    model1_out = TimeDistributed(Conv1D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_1'))(model1_in)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_01", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_1'))(model1_out)
    model1_out = TimeDistributed(Conv1D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_2'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_02", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_2'))(model1_out)
    model1_out = TimeDistributed(MaxPooling1D(pool_size=pool_size, name = 'times_maxp_1'))(model1_out)

    model1_out = TimeDistributed(Conv1D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_3'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_03", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_3'))(model1_out)
    model1_out = TimeDistributed(Conv1D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_4'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_04", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_4'))(model1_out)
    model1_out = TimeDistributed(MaxPooling1D(pool_size=pool_size, name = 'times_maxp_2'))(model1_out)

    model1_out = TimeDistributed(Flatten())(model1_out)
    model1_out = LSTM(50, activation=activation_dense, name = "times_lstm_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(unit_dense, activation=activation_dense, name = "times_dense_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(int(unit_dense/4), activation=activation_dense, name = "times_dense_2", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    return model1_in, model1_out

def conv2d_model(train_X_lstm, 
                 Normalize_method = "Batch", Normalize_rate = 0.01,
                 activation_conv = "relu", activation_dense = "relu", filters_conv = 64, kernel_conv = 2, pool_size=2,
                 unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    model1_out = Conv2D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_1')(model1_in)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_01")
    model1_out = Activation(activation_conv, name = 'times_act_1')(model1_out)
    model1_out = Conv2D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_2')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_02")
    model1_out = Activation(activation_conv, name = 'times_act_2')(model1_out)
    model1_out = MaxPooling2D(pool_size=pool_size, name = 'times_maxp_1')(model1_out)

    model1_out = Conv2D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_3')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_03")
    model1_out = Activation(activation_conv, name = 'times_act_3')(model1_out)
    model1_out = Conv2D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_4')(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_04")
    model1_out = Activation(activation_conv, name = 'times_act_4')(model1_out)
    model1_out = MaxPooling2D(pool_size=pool_size, name = 'times_maxp_2')(model1_out)

    model1_out = Flatten()(model1_out)
    model1_out = Dense(unit_dense, activation=activation_dense, name = "times_dense_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(int(unit_dense/4), activation=activation_dense, name = "times_dense_2", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    return model1_in, model1_out

def conv2d_lstm_model(train_X_lstm, 
                      Normalize_method = "Batch", Normalize_rate = 0.01,
                      activation_conv = "relu", activation_dense = "relu", filters_conv = 64, kernel_conv = 2, pool_size=2,
                      unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    model1_out = TimeDistributed(Conv2D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_1'))(model1_in)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_01", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_1'))(model1_out)
    model1_out = TimeDistributed(Conv2D(filters=filters_conv, kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_2'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_02", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_2'))(model1_out)
    model1_out = TimeDistributed(MaxPooling2D(pool_size=pool_size, name = 'times_maxp_1'))(model1_out)

    model1_out = TimeDistributed(Conv2D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_3'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_03", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_3'))(model1_out)
    model1_out = TimeDistributed(Conv2D(filters=int(filters_conv/2), kernel_size=kernel_conv, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate), name = 'times_conv_4'))(model1_out)
    model1_out = Normalize_f(model1_out, Normalize_method, Normalize_rate, Normalize_method+"_04", lstm = True)
    model1_out = TimeDistributed(Activation(activation_conv, name = 'times_act_4'))(model1_out)
    model1_out = TimeDistributed(MaxPooling2D(pool_size=pool_size, name = 'times_maxp_2'))(model1_out)

    model1_out = TimeDistributed(Flatten())(model1_out)
    model1_out = LSTM(50, activation=activation_dense, name = "times_lstm_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(unit_dense, activation=activation_dense, name = "times_dense_1", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    model1_out = Dense(int(unit_dense/4), activation=activation_dense, name = "times_dense_2", kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(model1_out)
    return model1_in, model1_out

def model_main_2(train_X, val_X,
                 l2_rate, activation_conv, activation_dense, optimizer, name, 
                 X_num_ = [train_X_num, val_X_num],
                 Y_train = banana_Y_train, Y_val = banana_Y_val):
    
    # create folder
    folder = 'model/' + name + "/" + str(l2_rate) + "-" + optimizer + "-" + activation_conv + "-" + activation_dense + "/"
    print(folder)
    
    try:
        os.mkdir('model/' + name)
    except:
        pass
    try:
        os.mkdir(folder)
    except:
        pass
    
    folder_new = 'model/' + name + "/"
    file_name_ = os.listdir(folder_new)
    file_exit_ = sum([[folder_new+x+"/" for y in os.listdir(folder_new+x) if 'final.h5' in y] for x in file_name_], [])
    
    if folder not in file_exit_:
        # dataset - classification
        #train_Y_1 = Y_train[[x for x in Y_train.columns if '????????????' in x]].values#.astype(int)
        #train_Y_1 = [train_Y_1[:,x] for x in range(train_Y_1.shape[1])]
        #val_Y_1 = Y_val[[x for x in Y_val.columns if '????????????' in x]].values#.astype(int)
        #val_Y_1 = [val_Y_1[:,x] for x in range(val_Y_1.shape[1])]
        # RepeatVector - regression
        train_Y_1 = Y_train[[x for x in Y_train.columns if 'RSV' in x]].values
        val_Y_1 = Y_val[[x for x in Y_val.columns if 'RSV' in x]].values
        # RepeatVector - sparse_categorical_crossentropy
        #train_Y_1 = Y_train[[x for x in Y_train.columns if '????????????' in x]].values
        #val_Y_1 = Y_val[[x for x in Y_val.columns if '????????????' in x]].values
        # RepeatVector - categorical_crossentropy
        #train_Y_1 = Y_train[[x for x in Y_train.columns if '????????????' in x]].values
        #train_Y_1 = to_categorical(train_Y_1, 3)
        #val_Y_1 = Y_val[[x for x in Y_val.columns if '????????????' in x]].values
        #val_Y_1 = to_categorical(val_Y_1, 3)
        
        # class weight
        # https://focal-loss.readthedocs.io/en/latest/generated/focal_loss.SparseCategoricalFocalLoss.html
        #class_weight_ = [class_weight_cal(x, banana_Y_train) for x in [x for x in banana_Y_train.columns if '????????????' in x]]
        #class_weight_ = {k:v for e in class_weight_ for (k,v) in e.items()}
        #class_weight_ = np.array([max([class_weight_[x][y] for x in class_weight_.keys()]) for y in [0,1,2]])
        # class_weight_ = [calculating_class_weights(train_Y_1[x]) for x in range(len(train_Y_1))]
        # RepeatVector - sparse_categorical_crossentropy
        #class_weight_ = [class_weight_cal(x, banana_Y_train) for x in [x for x in Y_train.columns if '????????????' in x]]
        #class_weight_ = {k:v for e in class_weight_ for (k,v) in e.items()}
        #class_weight_ = np.array([max([class_weight_[x][y] for x in class_weight_.keys()]) for y in [0,1,2]])

        # one-hot
        #train_Y_1 = [to_categorical(x, 3) for x in train_Y_1]
        #val_Y_1 = [to_categorical(x, 3) for x in val_Y_1]

        # build model
        if name == 'Conv1D': #train_X_lstm_1
            model1_in, model1_out = conv1d_model(train_X, 
                                                 Normalize_rate = l2_rate, 
                                                 activation_conv = activation_conv, 
                                                 activation_dense = activation_dense)
        if (name == 'Vanilla') | (name == 'Stacked') | (name == 'Bidirectional'): #train_X_lstm_1
            model1_in, model1_out = lstm_model(train_X, type_ = name, 
                                               Normalize_rate = l2_rate,
                                               activation_lstm = activation_conv, 
                                               activation_dense = activation_dense)
        if name == 'Conv1D_LSTM': #train_X_lstm_2
            model1_in, model1_out = conv1d_lstm_model(train_X, 
                                                      Normalize_rate = l2_rate,
                                                      activation_conv = activation_conv, 
                                                      activation_dense = activation_dense)
        if name == 'Conv2D_1': #train_X_lstm_2
            model1_in, model1_out = conv2d_model(train_X, 
                                                 Normalize_rate = l2_rate,
                                                 activation_conv = activation_conv, 
                                                 activation_dense = activation_dense)
        if name == 'Conv2D_2': #train_X_lstm_4
            model1_in, model1_out = conv2d_model(train_X, 
                                                 Normalize_rate = l2_rate,
                                                 activation_conv = activation_conv, 
                                                 activation_dense = activation_dense)
        if name == 'Conv2D_LSTM': #train_X_lstm_5
            model1_in, model1_out = conv2d_lstm_model(train_X, 
                                                      Normalize_rate = l2_rate,
                                                      activation_conv = activation_conv, 
                                                      activation_dense = activation_dense)

        model2_in, model2_out = non_time_series_model(X_num_[0], activation_ = activation_dense, l2_rate_ = l2_rate)
        concat = concatenate([model1_out, model2_out])
        out_1 = output_model_3(concat, 
                               #output_activation = 'softmax', 
                               output_activation = 'relu', 
                               activation_ = activation_dense, 
                               activation_1 = activation_conv, 
                               l2_rate = l2_rate)
        #out_1  = output_model_2(concat, output_activation = 'softmax', activation_ = activation_dense, l2_rate = l2_rate) 
        #softmax multiclass; sigmoid multilabel
        optimizer_use = optimizer_f(optimizer) 
        #callbacks_list = callbacks_mutilabel(folder) 
        callbacks_list = callbacks_regression(folder)
        merged_model = Model([model1_in, model2_in], out_1)
        merged_model.compile(#loss = SparseCategoricalFocalLoss(gamma = 0.5, class_weight = class_weight_),
                             #loss = "categorical_crossentropy",
                             #loss = weighted_categorical_crossentropy(class_weight_[0]),
                             #loss = {"day1": weighted_categorical_crossentropy(class_weight_[0]),
                             #        "day2": weighted_categorical_crossentropy(class_weight_[1]),
                             #        "day3": weighted_categorical_crossentropy(class_weight_[2]),
                             #        "day4": weighted_categorical_crossentropy(class_weight_[3]),
                             #        "day5": weighted_categorical_crossentropy(class_weight_[4])},
                             loss = 'mse',
                             #loss = "sparse_categorical_crossentropy",
                             #loss = "categorical_crossentropy",
                             optimizer = optimizer_use, 
                             metrics = ['mse'])

        # training model
        history = merged_model.fit([train_X, X_num_[0]], train_Y_1, 
                                   validation_data = ([val_X, X_num_[1]], val_Y_1), 
                                   callbacks = callbacks_list, #verbose = 0,
                                   #class_weight = class_weight_,
                                   epochs = 2000, batch_size = 64)

        # save final model
        for i in range(5):
        #while True:
            try:
                merged_model.save(folder + 'final.h5')
                break  
            except OSError:
                time.sleep(5) 

        # plot model summary
        plot_model(merged_model, to_file=  folder + 'model.png', show_shapes=True)

        # traiging fig & txt
        with open(folder + "history.txt",'w') as f:
            f.write(str(history.history))

        # draw training curve
        with open(folder + "history.txt") as f:
            lines = f.readlines()
        try:
            loss = list(map(lambda x:float(x), lines[0].split("], ")[0].split("[")[1].split(",")))
            accuracy = list(map(lambda x:float(x), lines[0].split("], ")[-3].split("[")[1].split(",")))
            val_loss = list(map(lambda x:float(x), lines[0].split("], ")[11].split("[")[1].split(",")))
            val_accuracy = list(map(lambda x:float(x), lines[0].split("], ")[-2].split("[")[1].split(",")))
        except:
            loss = list(map(lambda x:float(x), lines[0].split("], ")[0].split("[")[1].split(",")))
            accuracy = list(map(lambda x:float(x), lines[0].split("], ")[1].split("[")[1].split(",")))
            val_loss = list(map(lambda x:float(x), lines[0].split("], ")[2].split("[")[1].split(",")))
            val_accuracy = list(map(lambda x:float(x), lines[0].split("], ")[3].split("[")[1].split(",")))
        HH = [accuracy, val_accuracy, loss, val_loss]
        history_fig = show_train_history(HH)
        history_fig.savefig(folder + "history.png")
    else:
        print("pass")

def main(opt1):
    l2_rate = opt1['l2_rate']
    activation_conv = opt1['activation_conv']
    activation_dense = opt1['activation_dense']
    optimizer = opt1['optimizer']
    model_type = opt1['model_type']
    data_input = opt1['data_input']
    
    if data_input=='lstm_1':
        train_X = train_X_lstm_1
        val_X = val_X_lstm_1
    if data_input=='lstm_2':
        train_X = train_X_lstm_2
        val_X = val_X_lstm_2
    if data_input=='lstm_4':
        train_X = train_X_lstm_4
        val_X = val_X_lstm_4
    if data_input=='lstm_5':
        train_X = train_X_lstm_5
        val_X = val_X_lstm_5
        
    #print(data_input, model_type, l2_rate, activation_conv, activation_dense, optimizer)   
    
    model_main_2(train_X = train_X, 
                 val_X = val_X,
                 l2_rate = l2_rate, 
                 activation_conv = activation_conv, 
                 activation_dense = activation_dense, 
                 optimizer = optimizer, 
                 name = model_type)

if __name__ == "__main__":
    opt = parse_opt()
    for activation_conv in ["tanh", "swish", "leaky_relu", "gelu", "relu"]:
        for activation_dense in ["swish", "leaky_relu", "gelu", "relu"]:
            print(activation_conv, activation_dense)
            train_list = parameter(opt, activation_conv, activation_dense)
            pool = ThreadPool(len(train_list))
            pool.map(main, train_list)


