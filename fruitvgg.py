import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
from sklearn.metrics import confusion_matrix, classification_report

from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import multi_gpu_model

##############################################
learning_rate = 0.1  # initial learning rate
min_learning_rate = 0.00001  # once the learning rate reaches this value, do not decrease it further
learning_rate_reduction_factor = 0.5  # the factor used when reducing the learning rate -> learning_rate *= learning_rate_reduction_factor
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 2  # controls the amount of logging done during training and testing: 0 - none, 1 - reports metrics after each batch, 2 - reports metrics after each epoch
image_size = (100, 100)  # width and height of the used images
input_shape = (100, 100, 3)  # the expected input shape for the trained models; since the images in the Fruit-360 are 100 x 100 RGB images, this is the required input shape

use_label_file = False  # set this to true if you want load the label names from a file; uses the label_file defined below; the file should contain the names of the used labels, each label on a separate line
label_file = 'labels.txt'
base_dir = '/fruits-360_dataset/fruits-360'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test')
train_dir = os.path.join(base_dir, 'Training')
output_dir = '/vgg16/output_files'
##############################################

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]
else:
    labels = os.listdir(train_dir)
num_classes = len(labels)


print(labels)
print(num_classes)

validation_percent=0.1
batch_size=50

train_datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,  # randomly flip images
        validation_split=validation_percent)  # percentage indicating how much of the training set should be kept for validation

test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(train_dir, target_size=image_size, class_mode='categorical',
                                              batch_size=batch_size, shuffle=True, subset='training', classes=labels)
validation_gen = train_datagen.flow_from_directory(train_dir, target_size=image_size, class_mode='categorical',
                                                    batch_size=batch_size, shuffle=False, subset='validation', classes=labels)
test_gen = test_datagen.flow_from_directory(test_dir, target_size=image_size, class_mode='categorical',
                                                batch_size=batch_size, shuffle=False, subset=None, classes=labels)
                                                
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Sequential, Model

model_out_dir = os.path.join(output_dir, 'saved_weights')
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)

def image_process(x):
    import tensorflow as tf
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez

if K.image_data_format() == 'channels_first':
    input_shape = (3, 100, 100)
else:
    input_shape = (100, 100, 3)

#%% finetuning vgg16
img_input = Input(shape=input_shape, name='data')

base_model_vgg16 = VGG16(include_top=False, input_tensor=img_input, weights='imagenet')

#base_model_vgg16 = VGG16(include_top=False, input_tensor=img_input, weights=None)
#base_model_vgg16.load_weights('/home/sonali/mohit/models/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

model_vgg16=Sequential()
for l in base_model_vgg16.layers:
    model_vgg16.add(l)
for layer in model_vgg16.layers:
    layer.trainable = False

model_vgg16.add(Flatten())
model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(Dropout(0.5))
model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(Dropout(0.5))
model_vgg16.add(Dense(num_classes, activation='softmax'))

#parallel_model = multi_gpu_model(model_vgg16, gpus=2)

optimizer = Adadelta(lr=learning_rate)

model_vgg16.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#parallel_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=patience, verbose=verbose, 
                                            factor=learning_rate_reduction_factor, min_lr=min_learning_rate)
save_model = ModelCheckpoint(filepath=model_out_dir + "/model.h5", monitor='val_acc', verbose=verbose, 
                               save_best_only=True, save_weights_only=False, mode='max', period=1)
                               

epochs=100
history = model_vgg16.fit_generator(generator=train_gen,
                              epochs=epochs,
                              steps_per_epoch=(train_gen.n // batch_size) + 1,
                              validation_data=validation_gen,
                              validation_steps=(validation_gen.n // batch_size) + 1,
                              verbose=verbose,
                              callbacks=[learning_rate_reduction, save_model])

model_vgg16.load_weights(model_out_dir + "/model.h5")
validation_gen.reset()
loss_v, accuracy_v = model_vgg16.evaluate_generator(validation_gen, steps=(validation_gen.n // batch_size) + 1, verbose=verbose)
loss, accuracy = model_vgg16.evaluate_generator(test_gen, steps=(test_gen.n // batch_size) + 1, verbose=verbose)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss_v = %f" % (accuracy, loss))


model_vgg16=Sequential()
for l in base_model_vgg16.layers:
    model_vgg16.add(l)
for layer in model_vgg16.layers:
    layer.trainable = False

model_vgg16.add(Flatten())
model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(Dropout(0.5))
model_vgg16.add(Dense(4096, activation='relu'))
model_vgg16.add(Dropout(0.5))
model_vgg16.add(Dense(num_classes, activation='softmax'))

model_vgg16.load_weights(model_out_dir + "/model.h5")
model_vgg16.summary()

layer_outputs = [ model_vgg16.get_layer('dense_4').output, model_vgg16.get_layer('dense_5').output,model_vgg16.get_layer('block5_conv3').get_output_at(0)]

FC_layer_model = Model(inputs=model_vgg16.input,outputs=layer_outputs)

#validation_percent=0.1
train_datagen = ImageDataGenerator(
    width_shift_range=0.0,
    height_shift_range=0.0,
    zoom_range=0.0,
    horizontal_flip=True,
    vertical_flip=True)  # randomly flip images

test_datagen = ImageDataGenerator()

batch_size=50
image_size=(100, 100)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=image_size, 
                                              batch_size=batch_size, seed=1919)#, shuffle=False)
test_gen = test_datagen.flow_from_directory(test_dir, target_size=image_size, 
                                                batch_size=batch_size, seed=1919, shuffle=False)

epochs=10
i=0
train_features=np.zeros(shape=(epochs*train_gen.n, 26624))
train_labels=np.zeros(shape=(epochs*train_gen.n,1),dtype=int)
for x,y in train_gen:
    for a,b in zip(x,y):
      aa = image.img_to_array(a)
      aa = np.expand_dims(aa, axis=0)
      img = np.vstack([aa])
      FC_output = FC_layer_model.predict(img)
      train_features[i]=np.hstack((FC_output[0].flatten(), FC_output[1].flatten(), FC_output[2].flatten()))
      for bb in range(len(b)):
        if b[bb] != 0:
          train_labels[i]=bb
      i+=1
    if i == epochs*train_gen.n:
      break
      
i=0
test_features=np.zeros(shape=(test_gen.n, 26624))
test_labels=np.zeros(shape=(test_gen.n,1),dtype=int)
for x,y in test_gen:
    for a,b in zip(x,y):
      aa = image.img_to_array(a)
      aa = np.expand_dims(aa, axis=0)
      img = np.vstack([aa])
      FC_output = FC_layer_model.predict(img)
      test_features[i]=np.hstack((FC_output[0].flatten(), FC_output[1].flatten(), FC_output[2].flatten()))
      for bb in range(len(b)):
        if b[bb] != 0:
          test_labels[i]=bb
      i+=1
    if i == test_gen.n:
      break
      
      
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 250, random_state = 42, n_jobs = -1, max_features='auto')
rf = rf.fit(train_features, train_labels.ravel())

predictions = rf.predict(test_features)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(predictions , test_labels)
print(' Accuracy:', accuracy*100, '%.')

