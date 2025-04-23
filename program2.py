from keras import layers, models
from keras import optimizers
from keras import layers, models
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(10,activation="sigmoid")) #分類先の種類分設定

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

categories =[]
np.classes = len(categories)

X_train, X_test, y_train, y_test = np.load('')

X_train = X_train.astype('float') / 255
X_test  = X_test.astype('float')  / 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)

model = model.fit(X_train, y_train, 
                  epochs=10, batch_size=6,
                  validation_data=(X_test, y_test))