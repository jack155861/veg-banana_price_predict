from tensorflow.keras.layers import \
    BatchNormalization, Dropout, LayerNormalization, \
    Input, Conv1D, Activation, concatenate, Dense, Flatten, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, \
    LSTM, TimeDistributed, \
    Conv2D, MaxPool2D, GlobalMaxPool2D, MaxPooling2D, \
    PReLU, LeakyReLU, Bidirectional, ConvLSTM2D
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
import argparse
from multiprocessing.dummy import Pool as ThreadPool
# !pip install pydot
# !pip install graphviz 
# !sudo apt-get update
# !sudo apt install graphviz -y

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

banana_X_train = pd.read_pickle("banana_X_train.pkl")
banana_X_val = pd.read_pickle("banana_X_val.pkl")
banana_Y_train = pd.read_pickle("banana_Y_train.pkl")
banana_Y_val = pd.read_pickle("banana_Y_val.pkl")

cols_name = [[y+'-'+str(x).rjust(2,"0") for x in range(1,31)] for y in ['RSV','上價中位數','下價中位數','中價中位數','交易量','平均價','雨量']]
cols_name_2 = [x for x in banana_X_train.columns.tolist() if '-' not in x]
train_X_num = banana_X_train[cols_name_2[1:]].values
val_X_num = banana_X_val[cols_name_2[1:]].values

# shape (samples, 7 七個時間段, 7 七種特徵值)
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

# shape (samples, 7 七個時間段, 7 每個時間段往前推七個時間, 7 七種特徵值)
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

# shape (samples, 30 30個時間點, 30 每個時間點的數值為 0-30,  7 七種特徵值)
#plt.imshow(image_convert(banana_X_train[cols_name[0]].values[0])) 
#plt.show()
train_X_lstm_4 = np.stack([np.stack([image_convert(x) for x in banana_X_train[cols_name[y]].values]) for y in range(len(cols_name))], axis=3)
val_X_lstm_4 = np.stack([np.stack([image_convert(x) for x in banana_X_val[cols_name[y]].values]) for y in range(len(cols_name))], axis=3)

# shape (samples, 30 30個時間段, 30 每個時間段有連續累積的時間數據, 30 每個數據的數值為 0-30,  7 七種特徵值)
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

def swish(x, beta = 1):
    # https://www.geeksforgeeks.org/ml-swish-function-by-google-in-keras/
    return (x * sigmoid(beta * x))
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.25)
get_custom_objects().update({'swish': swish})
get_custom_objects().update({'leaky_relu': leaky_relu})

def optimizer_f(optimizer):
    if optimizer == "sgd":
        return SGD(learning_rate=0.001, momentum=0.9, decay=0, nesterov=True)
    if optimizer == "nadam":
        return Nadam(lr=0.001)
    if optimizer == 'adam':
        return Adam(learning_rate=0.001)

def callbacks_mutilabel(file_name, factor = 0.5):
    try:
        os.mkdir(file_name)
    except:
        pass
    
    # optimizers 
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=25, mode='min', min_delta=0.00001, 
                              cooldown=0, threshold_mode='rel', min_lr=1e-8, verbose = 2)    
    earlyst = EarlyStopping(monitor='val_loss', patience = 100, mode='min', min_delta=0.00001, verbose=2)    

    # save weight
    ck_callback_loss = ModelCheckpoint(file_name + '/weights_loss.hdf5',#_{epoch:04d}
                                       monitor="val_loss", mode='min', verbose=0, save_best_only=True, save_weights_only=False)
    ck_callback_loss._supports_tf_logs = False   
    ck_callback_acc = ModelCheckpoint(file_name + '/weights_accuracy.hdf5',#_{epoch:04d}
                                      monitor="val_binary_accuracy", mode='max', verbose=0, save_best_only=True, save_weights_only=False)
    ck_callback_acc._supports_tf_logs = False  
    
    ## https://stackoverflow.com/questions/58391418/how-to-use-custom-metric-from-a-callback-with-earlystopping-or-modelcheckpoint
    csv_callback = CSVLogger(file_name + '/log.csv', append=True, separator=',')
    
    return [ck_callback_loss, ck_callback_acc, csv_callback, lrate, earlyst]

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

def output_model_1(concat, output_activation, l2_rate, activation_ = 'relu'):
    out = Dense(1024, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(concat)
    out = Dropout(0.5)(out)
    out = Dense(256, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(64, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(5, activation=output_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    return out

def output_model_2(concat, output_activation, l2_rate, activation_ = 'relu'):
    out = Dense(1024, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(concat)
    out = Dropout(0.5)(out)
    out = Dense(256, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(64, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    out = Dense(1, activation=output_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_rate))(out)
    return out

def calculating_class_weights(y_true):
    # https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_true), y = y_true[:, i])
    return weights

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history[0])
    plt.plot(train_history[1])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
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

def lstm_model(train_X_lstm, type_, 
               Normalize_method = "Batch", Normalize_rate = 0.01,
               activation_lstm = "relu", activation_dense = "relu", unit_lstm = 100, unit_dense = 1024):
    model1_in = Input(shape=train_X_lstm.shape[1:])
    
    if type_=="Vanilla": #第一種方法Vanilla LSTM
        model1_out = LSTM(unit_lstm, activation=activation_lstm, name = 'Vanilla', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate))(model1_in) 
    if type_=="Stacked": #第二種方法Stacked LSTM
        model1_out = LSTM(unit_lstm, activation=activation_lstm, return_sequences=True, name = 'Stacked_1', kernel_initializer='he_normal', kernel_regularizer=l2(Normalize_rate))(model1_in) 
        model1_out = LSTM(unit_lstm, activation=activation_lstm, name = 'Stacked_2', kernel_initializer='he_normal')(model1_out) 
    if type_=="Bidirectional": #第三種方法Bidirectional LSTM
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

def model_main_1(train_X, val_X,
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
    file_exit_ = sum([[folder_new+x+"/" for y in os.listdir(folder_new+x) if 'history.png' in y] for x in file_name_], [])
    
    if folder not in file_exit_:
        # dataset
        train_Y_1 = Y_train[[x for x in Y_train.columns if '成本價格' in x]].values
        val_Y_1 = Y_val[[x for x in Y_val.columns if '成本價格' in x]].values

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
        out_1 = output_model_1(concat, output_activation = 'sigmoid', activation_ = activation_dense, l2_rate = l2_rate) #softmax multiclass; sigmoid multilabel
        optimizer_use = optimizer_f(optimizer) 
        callbacks_list = callbacks_mutilabel(folder)
        weight_input = calculating_class_weights(train_Y_1)
        merged_model = Model([model1_in, model2_in], out_1)
        merged_model.compile(loss = get_weighted_loss(weight_input), 
                             optimizer = optimizer_use, 
                             metrics = ["binary_accuracy"])

        # training model
        history = merged_model.fit([train_X,X_num_[0]], train_Y_1, 
                                   validation_data = ([val_X,X_num_[1]], val_Y_1), 
                                   callbacks = callbacks_list,
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
        
    print(data_input, model_type, l2_rate, activation_conv, activation_dense, optimizer)   
    
    model_main_1(train_X = train_X, 
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

