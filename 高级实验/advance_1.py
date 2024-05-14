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
    X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(label), test_size=0.20,
                                                        random_state=0)
    len_train = len(X_train)
    X_train_augmented = None
    y_train_augmented = None
    # 按照传入的百分比参数作进一步分割
    if (1 - percentage_of_data) != 0:
        X_train, X_test_useless, y_train, y_test_useless = train_test_split(X_train, y_train,
                                                                            test_size=(1 - percentage_of_data),
                                                                            random_state=0)
        # 添加数据增强
        datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            vertical_flip=False)
        datagen.fit(X_train)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 定义需要增强的数据数量
        augmented_data = []
        augmented_label = []
        num_augmented = int(len_train * (1 - percentage_of_data))  # 增强的数据数量，以使训练集的规模达到原来的规模
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=num_augmented):
            augmented_data.append(X_batch)
            augmented_label.append(y_batch)
            break  # 需要按需生成指定数量的数据，这里只取一个batch的数据

        augmented_data = np.concatenate(augmented_data)
        augmented_label = np.concatenate(augmented_label)

        # 将增强的数据合并到原数据中，形成新的训练集
        X_train_augmented = np.concatenate((X_train, np.array(augmented_data)))
        y_train_augmented = np.concatenate((y_train, np.array(augmented_label)))
    else:
        X_train_augmented = np.array(X_train)
        y_train_augmented = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # 将数据喂给模型
    history = model.fit(X_train_augmented,
                        y_train_augmented,
                        batch_size=64,
                        verbose=1,
                        epochs=(max_epoch - 1),
                        validation_data=(X_test, y_test),
                        shuffle=False
                        )

    print("After delete {:.2f}% data, Test_Accuracy: {:.2f}%".format(
        (1 - percentage_of_data),
        model.evaluate(np.array(X_test), np.array(y_test))[1] * 100))

    epoch_list = list(range(1, max_epoch))
    ax1.plot(epoch_list, history.history['val_accuracy'], label=str(percentage_of_data) + 'data Accuracy')
    ax2.plot(epoch_list, history.history['val_loss'], label=str(percentage_of_data) + 'data Loss')

f, (ax1, ax2) = plt.subplots(1, 2)
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
max_epoch = 5 + 1

plist = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
for p in plist:
    train_and_test(dataset, label, p, model_optimized, ax1, ax2, max_epoch)

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