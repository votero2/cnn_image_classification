import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

  # to calculate accuracy
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16



def flattening_Img():

  (X_train, y_train),(X_test, y_test) = mnist.load_data()
  # Flattening the images from the 28x28 pixels to 1D 787 pixels
  X_train = X_train.reshape(60000, 784)
  X_test = X_test.reshape(10000, 784)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  # normalizing the data to help with the training
  X_train /= 255
  X_test /= 255

  # one-hot encoding using keras' numpy-related utilities
  n_classes = 10
  print("Shape before one-hot encoding: ", y_train.shape)
  Y_train = to_categorical(y_train, n_classes)
  Y_test = to_categorical(y_test, n_classes)
  print("Shape after one-hot encoding: ", Y_train.shape)

  # building a linear stack of layers with the sequential model
  model = Sequential()
  # hidden layer
  model.add(Dense(100, input_shape=(784,), activation='relu'))
  # output layer
  model.add(Dense(10, activation='softmax'))

  # looking at the model summary
  model.summary()
  # compiling the sequential model
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
  # training the model for 10 epochs
  model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))


  






def calc_acuracy():
  # loading the dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # building the input vector from the 28x28 pixels
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
  X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  # normalizing the data to help with the training
  X_train /= 255
  X_test /= 255

  # one-hot encoding using keras' numpy-related utilities
  n_classes = 10
  print("Shape before one-hot encoding: ", y_train.shape)
  Y_train = to_categorical(y_train, n_classes)
  Y_test = to_categorical(y_test, n_classes)
  print("Shape after one-hot encoding: ", Y_train.shape)

  # building a linear stack of layers with the sequential model
  model = Sequential()
  # convolutional layer
  model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
  model.add(MaxPool2D(pool_size=(1,1)))
  # flatten output of conv
  model.add(Flatten())
  # hidden layer
  model.add(Dense(100, activation='relu'))
  # output layer
  model.add(Dense(10, activation='softmax'))

  # compiling the sequential model
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  # training the model for 10 epochs
  model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))







def cifar10_train():
    # keras imports for the dataset and building our neural network
    

    # loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # # building the input vector from the 32x32 pixels
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = to_categorical(y_train, n_classes)
    Y_test = to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()

    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))





#$ wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
#$ tar -xf imagenette2.tgz

imagenette_map = { 
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church",
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}







#model for img classification
def img_class():
   # create a new generator
   imagegen = ImageDataGenerator()
   # load train data
   train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
   # load val data
   val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

   # build a sequential model
   model = Sequential()
   model.add(InputLayer(input_shape=(224, 224, 3)))

   # 1st conv block
   model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
   model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
   # 2nd conv block
   model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
   model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
   model.add(BatchNormalization())
   # 3rd conv block
   model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
   model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
   model.add(BatchNormalization())
   # ANN block
   model.add(Flatten())
   model.add(Dense(units=100, activation='relu'))
   model.add(Dense(units=100, activation='relu'))
   model.add(Dropout(0.25))
   # output layer
   model.add(Dense(units=10, activation='softmax'))

   # compile model
   model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
   # fit on data for 30 epochs
   model.fit_generator(train, epochs=30, validation_data=val)






def cnn_model_for_img_class():
  imagegen = ImageDataGenerator()
   # load train data
  train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
   # load val data
  val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

  # build a sequential model
  model = Sequential()
  model.add(InputLayer(input_shape=(224, 224, 3)))

  # 1st conv block
  model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
  # 2nd conv block
  model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
  model.add(BatchNormalization())
  # 3rd conv block
  model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
  model.add(BatchNormalization())
  # ANN block
  model.add(Flatten())
  model.add(Dense(units=100, activation='relu'))
  model.add(Dense(units=100, activation='relu'))
  model.add(Dropout(0.25))
  # output layer
  model.add(Dense(units=10, activation='softmax'))

  # compile model
  model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
  # fit on data for 30 epochs
  model.fit_generator(train, epochs=30, validation_data=val)






def keras_app():
  imagegen = ImageDataGenerator()
   # load train data
  train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
   # load val data
  val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

 

  train_target = to_categorical(train.classes, num_classes=10)
  val_target = to_categorical(val.classes, num_classes=10)

  # include top should be False to remove the softmax layer
  pretrained_model = VGG16(include_top=False, weights='imagenet')
  pretrained_model.summary()

  # extract train and val features
  vgg_features_train = pretrained_model.predict(train)
  vgg_features_val= pretrained_model.predict(val)

  model2 = Sequential()
  model2.add(Flatten(input_shape=(7,7,512)))
  model2.add(Dense(100, activation='relu'))
  model2.add(Dropout(0.5))
  model2.add(BatchNormalization())
  model2.add(Dense(10, activation='softmax'))

  # compile the model
  model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

  model2.summary()

  # train model using features generated from VGG16 model
  model2.fit(vgg_features_train, train_target, epochs=50, batch_size=128, validation_data=(vgg_features_val, val_target))

    
if __name__ == "__main__":
    flattening_Img()
    calc_acuracy()
    cifar10_train()
    img_class()
    cnn_model_for_img_class()
    keras_app()



