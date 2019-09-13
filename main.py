import numpy
import pandas
import os
print(os.listdir("/Users/mariiaaksenova/PycharmProjects/untitled3/input/flower-rec/flower"))
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import SGD
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from subprocess import check_output
print(check_output(["ls", "/Users/mariiaaksenova/PycharmProjects/untitled3/input"]).decode("utf8"))


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3), activation='relu'))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(8,(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(256,activation='relu',kernel_initializer = 'uniform'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax',kernel_initializer = 'uniform'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.87, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('/Users/mariiaaksenova/PycharmProjects/untitled3/input/flower-rec/flower',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

early_stopping_callback = EarlyStopping(monitor='val_acc',patience=2)

history = model.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=25,
        shuffle = True,
        callbacks=[early_stopping_callback],
        )


