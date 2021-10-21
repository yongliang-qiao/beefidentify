"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D,ConvLSTM2D
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
from keras.regularizers import l2, l1
import sys
from keras import optimizers
#from resnet3d import Resnet3DBuilder
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
        elif model == 'DRNN':
            print("Loading DRNN model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.DRNN()
        elif model == 'Bilstm':
            print("Loading BiLSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.Bilstm()
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
            self.input_shape = (seq_length, 112, 112,3)
            self.model = self.c3d()
        elif model == 'c3dLSTM':
            print("Loading c3dLSTM")
            self.input_shape = (seq_length, 112, 112, 3)
            self.model = self.c3dLSTM()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        #optimizer = Adam(lr=1e-5, decay=1e-6)0.000001
        #optimizer = optimizers.SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #model.compile(optimizer=Nadam(lr=0.000001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',
        #optimizer = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-6)
        optimizer = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-6)
        #             metrics=['accuracy'])
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


    def DRNN(self):

        model = Sequential()
    # This layer converts frequency space to hidden space
        model.add(TimeDistributed(Dense(200), input_shape=self.input_shape))
        model.add(TimeDistributed(Dense(200)))
        model.add(TimeDistributed(Dense(200)))
        for cur_unit in range(2):
          print('Creating LSTM with %s neurons' % (2048))
          model.add(LSTM(200, return_sequences=True, recurrent_regularizer=l2(0)))
            #model.add(LSTM(num_hidden_dimensions, return_sequences=True))
          model.add(BatchNormalization())

        myoutput = int(self.input_shape[1] / 2)
        model.add(TimeDistributed(Dense(myoutput, activation='relu')))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model
        #######kernel_regularizer=l2(0.05),activity_regularizer=l1(0.05),
    def Bilstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        # model.add(Bidirectional(LSTM(2048, return_sequences=True),input_shape=self.input_shape))
        # model.add(Bidirectional(LSTM(2048)))  id identification is 2048
        #model.add(Bidirectional(LSTM(2048, return_sequences=True), input_shape=self.input_shape))
        model.add(Bidirectional(LSTM(2048, return_sequences=True), input_shape=self.input_shape))
        #model.add(Bidirectional(LSTM(2048)))
       # model.add(BatchNormalization())#######dairy
        #model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Flatten())
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
        model.add(LSTM(1024, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(4096))
        model.add(Dropout(0.6))
        model.add(Dense(4096))
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
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(2045, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        #model.add(Dense(2048, activation='relu', name='fc7'))
       # model.add(Dropout(0.5))
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
        model.add(ConvLSTM2D(256, kernel_size=(3, 3), activation='relu', padding='valid', return_sequences=True,kernel_regularizer=l2(0.1),recurrent_dropout=0.5))
        model.add(BatchNormalization())
        model.add(Flatten())
        #new added two dense
        model.add(Dense(2048, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        # FC layers group
        #x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
        #     x = Dropout(0.5)(x)
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
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