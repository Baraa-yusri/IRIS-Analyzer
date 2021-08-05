import os
import gc
import cv2
import random
from collections import defaultdict
from itertools import chain

import keras.initializers
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# 1.) Get image path set

classes = defaultdict(list)
for filename in os.listdir("CroppedImages"):
    userId = filename.split('_')[0]  # split the filename and retain only the first element which contains identity
    listname = "user_" + userId[1:]  # listname = user + identity without 'u' character
    classes[listname].append('CroppedImages/' + filename)

# 2.) Shuffling, Training and Testing Image Sets
train_set = []
test_set = []

for key in classes:
    random.shuffle(classes[key])  # shuffle so there is no  bias between training & testing set selection
    train_set.append(classes[key][:(len(classes[key]) // 2)])  # we use half as our training set
    test_set.append(classes[key][(len(classes[key]) // 2):])  # the other half as our test set

# un-list the sub-lists to make only one big list
train_set = list(chain.from_iterable(train_set))
test_set = list(chain.from_iterable(test_set))

# 4.) Garbage Collection
del classes
gc.collect()

# 5.) Image Pre-Processing - reformat images so that model knows what dimensions to expect
nRows = 150  # Width
nCols = 150  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1

# 6.) Training and Testing Set Labeling - we are going to use these arrays to contain the read images along with their label.
X_train = []
X_test = []
y_train = []
y_test = []

# 7.) Read and Label Each Image in the Training Set
for image in train_set:
    try:
        # read in and resize the image based on our static dimensions from 5 above
        X_train.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))

        y_train.append(int(userId[1:]))


    except Exception:  # images failed to be read
        print('Failed to format: ', image)

# 8.) Read and Label Each Image in the Testing Set
for image in test_set:
    try:
        # read in and resize the image based on our static dimensions from 5 above
        X_test.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))

        # to determine the class of each read image.
        userId = image.split('/')[1]
        userId = userId.split('_')[0]
        y_test.append(int(userId[1:]))

    except Exception:  # images failed to be read
        print('Failed to format: ', image)

# 9.) Garbage Collection
del train_set, test_set
gc.collect()

# 10.) Convert to Numpy Arrays so that it can be fed to CNN, and reformat the input and target data using accompanying libraries like Scikit-learn and Keras.
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 11.) Switch Targets to Categorical - So that we can use a softmax activation function X ∈ [0, 1] to predict the image class we are going to convert our vector of labels where L ∈ {14 - 154} to a categorical set L ∈ {0, 1}.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 12.) Convolutional Neural Network
init = keras.initializers.HeNormal(seed=None)

model = Sequential()
model.add(Conv2D(32, kernel_size=7, kernel_initializer=init, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))  # down-sample input
model.add(Conv2D(64, kernel_size=3, kernel_initializer=init, activation='relu'))
model.add(Conv2D(64, kernel_size=3, kernel_initializer=init, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, kernel_size=11, kernel_initializer=init, activation='relu'))
model.add(Conv2D(256, kernel_size=11, kernel_initializer=init, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, kernel_size=3, kernel_initializer=init, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(256, kernel_size=5, kernel_initializer=init, activation='relu'))
model.add(Conv2D(512, kernel_size=5, kernel_initializer=init, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, kernel_size=3, kernel_initializer=init, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())  # prepares a vector for the fully connected layers
model.add(Dropout(0.5))  #
model.add(Dense(101, activation='softmax'))

# 13.) Model Summary
print(model.summary())
# ------------------------------------------------------Save model----------------------------------------------
checkpoint_path = "identification_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(X_train, y_train, validation_data=(X_train[:20], y_train[:20]), epochs=500,
          callbacks=[cp_callback])  # Pass callback to training

_, accuracy = model.evaluate(X_test)

print(accuracy)
# ---------------------------------------------------------------------------------------------------------------------

# 15.) Plot Accuracy Over Training Period
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# 16.) Plot Loss Over Training Period
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
