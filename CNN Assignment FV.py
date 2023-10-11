#!/usr/bin/env python
# coding: utf-8

# Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer
# that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images 
# and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# In[1]:


#importing the libraries
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


# Defining the path for train and test images
## Todo: Update the paths of the train and test dataset 
data_dir_train = pathlib.Path("C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Train")
data_dir_test = pathlib.Path("C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Test")


# In[3]:


#to count the images in each folder train and test
image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
print(image_count_train)
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print(image_count_test)


# ### Load using keras.preprocessing
# Let's load these images off disk using the helpful image_dataset_from_directory utility.

# ## Create a dataset
# Define some parameters for the loader:

# In[4]:


batch_size = 32
img_height = 180
img_width = 180


# In[6]:


## Write your train dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    seed=123,                # Set the seed
    image_size=(img_height, img_width),  # Resize images to the specified dimensions
    batch_size=32,            # Batch size
    shuffle=True               # Shuffle the dataset
)


# In[7]:


## Write your validation dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    seed=123,                # Set the seed
    image_size=(img_height, img_width),  # Resize images to the specified dimensions
    batch_size=32,            # Batch size
    shuffle=True               # Shuffle the dataset
)


# In[8]:


# List out all the classes of skin cancer and store them in a list. 
# You can find the class names in the class_names attribute on these datasets. 
# These correspond to the directory names in alphabetical order.
class_names = train_ds.class_names
print(class_names)


# ## Visualize the data
# Todo, create a code to visualize one instance of all the nine classes present in the dataset

# In[9]:


import matplotlib.pyplot as plt

def visualize_dataset(train_ds, num_samples=9):
    plt.figure(figsize=(12, 12))
    for images, labels in train_ds.take(1):
        for i in range(num_samples):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
    plt.show()
    
    
visualize_dataset(train_ds)


# In[10]:


def visualize_data(val_ds, num_samples=9):
    plt.figure(figsize=(12, 12))
    for images, labels in val_ds.take(1):
        for i in range(num_samples):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
    plt.show()
    
    
visualize_data(val_ds)


# In[11]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[12]:


from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
input_shape = (180, 180, 3)
num_classes=9
model = Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255, input_shape=input_shape))

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[13]:


# summary of the model
print(model.summary())


# In[14]:


### Todo, choose an appropirate optimiser and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use appropriate loss function
              metrics=['accuracy'])  # Add any desired metrics


# In[15]:


# summary
model.summary()


# In[16]:


#train the model
### from tensorflow.keras.utils import to_categorical
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
    shuffle= True,
  epochs=epochs,
    batch_size=32
)


# In[17]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[19]:


import tensorflow as tf

# Load your training dataset using tf.keras.utils.image_dataset_from_directory
data_dir_train = pathlib.Path("C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Train") 
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir_train)

# Initialize an empty dictionary to store class counts
class_counts = {}

# Count the occurrences of each class in the training dataset
for _, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] = class_counts.get(label, 0) + 1

# Print the class distribution
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")


# In[20]:


import tensorflow as tf

# Load your training dataset using tf.keras.utils.image_dataset_from_directory
data_dir_train = pathlib.Path("C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Train") 
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir_train)

# Initialize an empty dictionary to store class counts
class_counts = {}

# Count the occurrences of each class in the training dataset
total_samples = 0
for _, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] = class_counts.get(label, 0) + 1
        total_samples += 1

# Calculate and print the class proportions
print("Class proportions in the training dataset:")
for label, count in class_counts.items():
    proportion = count / total_samples
    print(f"Class {label}: {count} samples ({proportion * 100:.2f}%)")


# ## Results
# We noted that data is underfiting so in next step we will augment the data and then again train the model. We saw that class 6 has the least number of classes. Further we noted that class 5 and Class 3 dominate the data in terms proportionate number of samples.

# ## Data Augmentation

# In[21]:


get_ipython().system('pip install Augmentor')


# In[23]:


path_to_training_dataset = "C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Train"
import Augmentor
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + '/' + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500) ## We are adding 500 samples per class to make sure that none of the classes are sparse.


# In[24]:


image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)


# ## Lets see the distribution of augmented data after adding new images to the original training data

# In[28]:


from glob import glob
path_list = [x for x in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
path_list


# In[29]:


lesion_list_new = [os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
lesion_list_new


# In[31]:


dataframe_dict_new = dict(zip(path_list, lesion_list_new))


# In[33]:


batch_size = 32
img_height = 180
img_width = 180


# In[37]:


data_dir_train="C:/Users/SHWETA/Dropbox/Upgrad course/Neural networks/Train"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    seed=123, 
    validation_split = 0.2,
    subset = 'training',
    image_size=(img_height, img_width),  # Resize images to the specified dimensions
    batch_size=32,            # Batch size
    shuffle=True               # Shuffle the dataset
)


# In[38]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = 'validation',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[39]:


from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
input_shape = (180, 180, 3)
num_classes=9
model = Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1.0 / 255, input_shape=input_shape))

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[40]:


# summary of the model
print(model.summary())


# In[41]:


### Todo, choose an appropirate optimiser and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use appropriate loss function
              metrics=['accuracy'])  # Add any desired metrics


# In[ ]:


#train the model
### from tensorflow.keras.utils import to_categorical
epochs = 30
history = model.fit(
  train_ds,
  validation_data=val_ds,
    shuffle= True,
  epochs=epochs,
    batch_size=32
)


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## yes, rebalancing helped the data and data fitted well.

# In[ ]:




