#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

import numpy as np
np.random.seed(1000)
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

os.environ['KERAS_BACKEND'] = 'tensorflow'

# Iterate through all images in Parasitized folder, resize to 64 x 64
# Then save as numpy array with name 'dataset'
# Set the label to this as 0
data_root_dir = r'C:\Users\plotnickslope\mlExperiments\\'  # 改成你自己的数据存放目录！
image_directory = data_root_dir + 'cell_images\\'
SIZE = 64
dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
label = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

parasitized_images = os.listdir(image_directory + 'Parasitized\\')
for i, image_name in enumerate(
        parasitized_images):  # Remember enumerate method adds a counter and returns the enumerate object

    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Parasitized\\' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)
print('parasitized_images have been read!')

# Iterate through all images in Uninfected folder, resize to 64 x 64
# Then save into the same numpy array 'dataset' but with label 1

uninfected_images = os.listdir(image_directory + 'Uninfected\\')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Uninfected\\' + image_name)
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)
print('uninfected_images have been read!')

# Apply CNN

# Optimized model
inputShape = (64, 64, 3)
model_optimized = Sequential()
model_optimized.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
model_optimized.add(MaxPooling2D(2, 2))
model_optimized.add(BatchNormalization(axis=-1))
model_optimized.add(Dropout(0.2))

model_optimized.add(Conv2D(32, (3, 3), activation='relu'))
model_optimized.add(MaxPooling2D(2, 2))
model_optimized.add(BatchNormalization(axis=-1))
model_optimized.add(Dropout(0.2))

model_optimized.add(Conv2D(32, (3, 3), activation='relu'))
model_optimized.add(MaxPooling2D(2, 2))
model_optimized.add(BatchNormalization(axis=-1))
model_optimized.add(Dropout(0.2))

model_optimized.add(Flatten())

model_optimized.add(Dense(512, activation='relu'))
model_optimized.add(BatchNormalization(axis=-1))
model_optimized.add(Dropout(0.5))
model_optimized.add(Dense(2, activation='softmax'))

model_optimized.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])

print(model_optimized.summary())
###############################################################

def train_and_test(dataset, label, percentage_of_data, model, ax1, ax2, max_epoch):
    # initialize
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20,
                                                        random_state=0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    N = len(X_train)
    idx_labeled = np.random.choice(N, int(percentage_of_data * N), replace=False)
    idx_unlabeled = np.setdiff1d(np.arange(N), idx_labeled)
    X_labeled = X_train[idx_labeled]
    X_unlabeled = X_train[idx_unlabeled]
    y_labeled = y_train[idx_labeled]


    left_epoch = 5
    # training loop
    for epoch in range(1, max_epoch):
        left_epoch = left_epoch - 1
        print("Epoch %d/%d" % (epoch, max_epoch - 1))
        model.fit(X_labeled, y_labeled, epochs=1, validation_data=(X_test, y_test), batch_size=64, verbose=1, shuffle=False)
        if len(X_unlabeled) == 0:
            break   # we have used all unlabeled samples
        proba_unlabeled = model.predict(X_unlabeled)
        max_proba_unlabeled = proba_unlabeled.max(axis=1)
        reliable_idx = np.where(max_proba_unlabeled > 0.9)[0]   # high confidence
        if len(reliable_idx) > 0:
            X_labeled = np.concatenate([X_labeled, X_unlabeled[reliable_idx]], axis=0)
            y_labeled = np.concatenate([y_labeled, to_categorical(proba_unlabeled[reliable_idx].argmax(axis=1), num_classes=y_labeled.shape[1])], axis=0)
            X_unlabeled = np.delete(X_unlabeled, reliable_idx, axis=0)

    if left_epoch > 0:
        print("Left Training Start!")
        model.fit(X_labeled,
                    y_labeled,
                    batch_size=64,
                    verbose=1,
                    epochs=left_epoch,
                    validation_data=(X_test, y_test),
                    shuffle=False
                    )

    print("After delete {:.2f}% data's label, Test_Accuracy: {:.2f}%".format(
        (1 - percentage_of_data),
        model.evaluate(np.array(X_test), np.array(y_test))[1] * 100))

f, (ax1, ax2) = plt.subplots(1, 2)
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
max_epoch = 5 + 1

plist = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
for p in plist:
    train_and_test(np.array(dataset), np.array(label), p, model_optimized, ax1, ax2, max_epoch)

ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

plt.show()  # show the pictures

# Save the model
model_optimized.save('malaria_cnn_optimized.h5')