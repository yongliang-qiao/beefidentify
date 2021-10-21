from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Flatten, GRU, Dense, Dropout
from keras import optimizers

def build_model():
    pretrained_cnn = VGG16(weights='imagenet', include_top=False)
    # pretrained_cnn.trainable = False
    for layer in pretrained_cnn.layers[:-5]:
        layer.trainable = False
    # input shape required by pretrained_cnn
    input = Input(shape = (224, 224, 3)) 
    x = pretrained_cnn(input)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    pretrained_cnn = Model(inputs = input, output = x)

    input_shape = (None, 224, 224, 3) # (seq_len, width, height, channel)
    model = Sequential()
    model.add(TimeDistributed(pretrained_cnn, input_shape=input_shape))
    model.add(GRU(1024, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(categories, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
                metrics=['accuracy'])
    return model


import numpy as np
import tensorflow as tf
from keras import layers, models, applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda
import cv2
from keras import backend as K
from keras.utils import plot_model

# set learning phase to 0
K.set_learning_phase(0)


video = layers.Input(shape=(None, 299,299,3),name='video_input')
cnn = applications.inception_v3.InceptionV3(
    weights='imagenet',
    include_top=False,
    pooling='avg')
cnn.trainable = False
# wrap cnn into Lambda and pass it into TimeDistributed
encoded_frame = layers.TimeDistributed(Lambda(lambda x: cnn(x)))(video)
encoded_vid = layers.LSTM(256)(encoded_frame)
outputs = layers.Dense(128, activation='relu')(encoded_vid)
model = models.Model(inputs=[video],outputs=outputs)
model.compile(optimizer='adam',loss='mean_squared_logarithmic_error')

# plot_model(model, to_file='model.png')

# model.summary()

# Generate random targets
y = np.random.random(size=(128,)) 
y = np.reshape(y,(-1,128))
frame_sequence = np.random.random(size=(1, 48, 299, 299, 3))
model.fit(x=frame_sequence, y=y, validation_split=0.0,shuffle=False, batch_size=1, epochs=5)






