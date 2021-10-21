"""
A collection of models we'll use to attempt to classify videos.
"""
#from keras_self_attention import SeqSelfAttention
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, multiply,  Activation, \
    RepeatVector, Permute,  merge, Lambda, Reshape, concatenate
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from collections import deque
from keras.regularizers import l2, l1
import sys
from keras import optimizers
from keras.layers import Dense
#from keras import Input, Model
#from tensorflow.keras import Input
#from tcn import TCN
from keras.models import Sequential
from keras import backend as K
#K.set_image_dim_ordering('tf')
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adadelta, Adam
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
#from attention_utils import get_activations, get_data_recurrent
#from resnet3d import Resnet3DBuilder
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.num_input_tokens = None
        #self.nb_classes = None
        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        #if self.nb_classes >= 10:
         #   metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lstmdouble':
            print("Loading LSTMdouble model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstmdouble()
        elif model == 'SimpleRNN':
            print("Loading SimpleRNN model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.SimpleRNN()
        elif model == 'cnn_attention_lstm':
            print("Loading cnn_attention_lstm model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'cnn_lstm_attention':
            print("Loading cnn_lstm_attention model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 112, 112, 3)
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 112, 112 ,3)
            self.model = self.conv_3d()
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length,  112, 112,3)
            self.model = self.c3d()
        elif model == 'c3dLSTM':
            print("Loading c3dLSTM")
            self.input_shape = (seq_length, 112, 112, 3)
            self.model = self.c3dLSTM()
       # elif model == 'TCN':
       #     print("Loading TCN")
        elif model == 'Bilstm':
            print("Loading Bilstm")
            self.input_shape = (seq_length, features_length)
            self.model = self.Bilstm()
            # self.input_shape = (seq_length, 112, 112, 3)
            # self.model = self.c3d() Bilstm
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        #optimizer = Adam(lr=1e-5, decay=1e-6)0.000001  0.000001
        #optimizer = optimizers.SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = optimizers.SGD(lr=0.000005, decay=1e-6, momentum=0.9, nesterov=True)
       ##################### optimizer=optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False,decay=1e-6)
        #optimizer = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #model.compile(optimizer=Nadam(lr=0.000001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',
        #             metrics=['accuracy'])
        #optimizer = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-6)
        optimizer = optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-6)
        #optimizer = tf.keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=['accuracy'])
        print(self.model.summary())
    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.0))
        #model.add(Flatten()) #qiao_added
        # model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model
    def lstmdouble(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.0))
        model.add(LSTM(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.0))
        #model.add(Flatten()) #qiao_added
        # model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model


    def SimpleRNN(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(SimpleRNN(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.0))
        #model.add(Flatten()) #qiao_added
        # model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def cnn_attention_lstm2(self):
        model = Sequential()
        model.add(attention_block(self.input_shape, self.num_input_tokens))
        model.add(LSTM(2048, return_sequences=False))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        # inputs = Input(shape=(self.input_shape, self.num_input_tokens,))
        # attention_inputs = attention_block(inputs, self.input_shape)
        # #lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=False))(attention_inputs)
        # lstm_out = LSTM(2048, return_sequences=False)(attention_inputs)
        # x = Dense(512, activation='relu')(lstm_out)
        # x = Dropout(0.5)(x)
        # x = Dense(self.nb_classes, activation='softmax')(x)
        # model = Model(input=[inputs], output=x)
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #print(model.summary())
        return model

    def cnn_attention_lstm(self):
        inputs = Input(shape=(self.input_shape, self.num_input_tokens,))
        attention_inputs = attention_block(inputs, self.input_shape)
        #lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=False))(attention_inputs)
        lstm_out = LSTM(2048, return_sequences=False)(attention_inputs)
        x = Dense(512, activation='relu')(lstm_out)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #print(model.summary())
        return model

    def cnn_lstm_attention(self):
        inputs = Input(shape=(self.input_shape, self.num_input_tokens,))
        #lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
        lstm_out = LSTM(1024, return_sequences=True,dropout=0.5)(inputs)
        attention_mul = attention_block(lstm_out, self.input_shape)
        attention_mul = Flatten()(attention_mul)
        x = Dense(512, activation='relu')(attention_mul)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(input=[inputs], output=x)
        #model.compile(loss='categorical_crossentropy', optimizer='
        #
        # metrics=['accuracy'])
        print(model.summary())
        return model


    def lstm_atten(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Flatten()) #qiao_added
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.5))

        attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(2048)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = concatenate([activations, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2048,))(sent_representation)

        probabilities = Dense(self.nb_classes, activation='softmax')(sent_representation)

        model = model(input=self.input_shape, output=probabilities )

        dense1800 = Dense(4096, activation='relu')

        #dense1800 = Dense(1800, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        attention_probs = Dense(4096, activation='sigmoid', name='attention_probs')(dense1800)
        attention_mul = multiply([dense1800, attention_probs], name='attention_mul')
        dense7 = Dense(self.nb_classes, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(attention_mul)
        model = Model(input=[self.input_shape], output=dense7)
        return model

   # def TCN(self):
        # https://github.com/philipperemy/keras-tcn
    #    batch_size = 20
    #    features_length = 2048
    #    i = Input(batch_shape=(batch_size, self.seq_length, features_length))

    #    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    #    o = Dense(self.nb_classes, activation='softmax')(o)

    #    model = Model(inputs=[i], outputs=[o])
    #    return model

    def Bilstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        # model.add(Bidirectional(LSTM(2048, return_sequences=True),input_shape=self.input_shape))
        # model.add(Bidirectional(LSTM(2048)))  id identification is 2048
        model.add(Bidirectional(LSTM(2048, return_sequences=True), input_shape=self.input_shape))
        #model.add(Bidirectional(LSTM(2048)))
        model.add(Dense(2048, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        # model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
        # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        # model = Sequential()
        # model = Sequential()
        # model.add(Embedding(max_features, 128, input_length=maxlen))
        # model.add(Bidirectional(LSTM(64)))
        # model.add(Dropout(0.5))
        # model.add(Dense(1, activation='sigmoid'))

        # model.add(Embedding(20000, 128, input_length=self.seq_length))
        # model.add(Flatten(input_shape=self.input_shape))
        # model.add(Embedding(20000, 128, input_length=self.seq_length))
        # model.add(Bidirectional(LSTM(128)))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.nb_classes, activation='softmax'))
        return model
#######kernel_regularizer=l2(0.05),activity_regularizer=l1(0.05),
    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        #model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(LSTM(4096, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(1024))
        model.add(Dropout(0.6))
        model.add(Dense(512))
        model.add(Dropout(0.6))
        model.add(Dense(self.nb_classes,  activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
        #                        border_mode='valid', name='pool5', dim_ordering="tf"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='same', name='pool5', dim_ordering="tf"))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        for layer in model.layers:
            print(layer.output_shape)
        return model
    def c3dLSTM(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='same', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='same', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='same', name='pool3'))
        # 4th layer groupprint(layer.get_shape())
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='same', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='same', name='pool5'))
        shaped=model.layers[13].output.get_shape()
        print(shaped[1], shaped[2], shaped[3])
        # model.add(ConvLSTM2D(512, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=[self.seq_length,shaped[1],shaped[2],shaped[3]],
        #                     return_sequences=True)) GlobalAveragePooling3D
       # model.add(ConvLSTM2D(512, kernel_size=(3, 3), activation='relu', padding='valid',
        #                     input_shape=[1, shaped[1], shaped[2], shaped[3]],
         #                    return_sequences=True,kernel_regularizer=l2(0.1),recurrent_dropout=0.5))
        model.add(ConvLSTM2D(512, kernel_size=(3, 3), activation='relu', padding='valid', return_sequences=True,kernel_regularizer=l2(0.1),recurrent_dropout=0.5))
        model.add(ConvLSTM2D(640, kernel_size=(3, 3), activation='relu', padding='valid', return_sequences=True,
                             kernel_regularizer=l2(0.1), recurrent_dropout=0.5))
        model.add(BatchNormalization())
        model.add(Flatten())
        #new added two dense
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        # FC layers group
        #x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
        #     x = Dropout(0.5)(x)
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    # model.add(LSTM(2048, return_sequences=True,
    #                input_shape=self.input_shape,
    #                dropout=0.5))
    # model.add(Flatten())  # qiao_added
    # # model.add(Dense(1024, activation='relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.0))
    # model.add(Dense(self.nb_classes, activation='softmax'))

    # def c3d_model():
    #     input_shape = (112, 112, 16, 3)
    #     weight_decay = 0.005
    #     nb_classes = 101
    #
    #     inputs = Input(input_shape)
    #     x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #                activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    #     x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    #
    #     x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #                activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    #
    #     x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #                activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    #
    #     x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #                activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    #
    #     x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
    #                activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    #
    #     x = Flatten()(x)
    #     x = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = Dropout(0.5)(x)
    #     x = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    #     x = Dropout(0.5)(x)
    #     x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
    #     x = Activation('softmax')(x)
    #
    #     model = Model(inputs, x)


    # model = Resnet3DBuilder.build_resnet_50((96, 96, 96, 1), 20)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(X_train, y_train, batch_size=32)


    ####https://github.com/pantheon5100/3D-CNN-resnet-keras
