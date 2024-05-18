# Import all required modules
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from matplotlib.image import imread
import zipfile
import matplotlib.pyplot as plt
import glob
from random import random, seed
from shutil import copyfile
import pandas as pd
import seaborn as sns

# Extract Image data from zip files
def extract_data(input_train_dir, input_test_dir, output_dir):
    with zipfile.ZipFile(input_train_dir, 'r') as zipp:
        zipp.extractall(output_dir)
    with zipfile.ZipFile(input_test_dir, 'r') as zipp:
        zipp.extractall(output_dir)

# Visualize Images
def visualize_data(dir):
    plt.figure(figsize=(10,3)) # specifying the overall grid size
    plt.subplots_adjust(hspace=0.4)
    images = glob.glob(dir)
    for i in range(10):
        plt.subplot(1,10,i+1)    # the number of images in the grid is 10*10 (100)
        img = images[i]
        label = (os.path.basename(img))[:3]
        image = imread(img)
        plt.imshow(image)
        plt.title(label,fontsize=12)
        plt.axis('off')
    plt.show()

# Train-Val-Test Split
def train_val_split(src_dir, dest_dir, val_ratio):

    subdirs = ['train/', 'val/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['dogs/', 'cats/']
        for labldir in labeldirs:
            newdir = dest_dir + subdir + labldir
            os.makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)
    # copy training dataset images into subdirectories
    for file in os.listdir(src_dir):
        src = src_dir + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'val/'
        if file.startswith('cat'):
            dst = dest_dir + dst_dir + 'cats/' + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dest_dir + dst_dir + 'dogs/' + file
            copyfile(src, dst)

    path1 = "Dataset/train/cats"
    path2 = "Dataset/train/dogs"
    path3 = "Dataset/val/cats"
    path4 = "Dataset/val/dogs"
    print('Then number of cat images in training data is' ,len(os.listdir(path1)))
    print('Then number of dog images in training data is' ,len(os.listdir(path2)))
    print('Then number of cat images in validation data is' ,len(os.listdir(path3)))
    print('Then number of dog images in validation data is' ,len(os.listdir(path4)))

def model_design(image_size, image_channel):

    model = Sequential()
    # Input Layer. Receives input images for classification.
    model.add(Conv2D(32,(3,3),activation='relu',input_shape = (image_size,image_size,image_channel))) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Block 1 
    model.add(Conv2D(64,(3,3),activation='relu'))  # Extract features from the images through convolutional operations.
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2))) # Reduce the spatial dimensions of the feature maps.
    model.add(Dropout(0.2))
    # Block 2
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Block 3
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # Fully Connected layers. Perform classification using densely connected layers.
    model.add(Flatten())   # Convert the 2D feature maps into a 1D vector.
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(1,activation='sigmoid'))  # Provides the final prediction probabilities for cat and dog classes.
    return model

# Creating image data generator with augmentation features
def image_generator(image_size, batch_size, image_dir, train_status):
    
    if train_status==True:
        datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range = 15,
                                            horizontal_flip = True,
                                            zoom_range = 0.2,
                                            shear_range = 0.1,
                                            fill_mode = 'reflect',
                                            width_shift_range = 0.1,
                                            height_shift_range = 0.1)
        gen = datagen.flow_from_directory(image_dir, 
                                            class_mode='binary',
                                            target_size = (image_size,image_size),
                                            batch_size = batch_size,
                                            shuffle = True
                                            )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_directory(image_dir, 
                                                class_mode='binary',
                                                batch_size = batch_size,
                                                target_size = (image_size,image_size),
                                                shuffle = False
                                                )
    return gen


# plots for accuracy and Loss with epochs
def generate_plots(model_hist):
    
    error = pd.DataFrame(model_hist)
    print(error.head())
    plt.figure(figsize=(18,5),dpi=200)
    sns.set_style('darkgrid')

    plt.subplot(121)
    plt.title('Cross Entropy Loss',fontsize=15)
    plt.xlabel('Epochs',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.plot(error['loss'], label='train_loss')
    plt.plot(error['val_loss'], label='val_loss')
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.title('Accuracy',fontsize=15)
    plt.xlabel('Epochs',fontsize=12)
    plt.ylabel('Accuracy',fontsize=12)
    plt.plot(error['accuracy']*100, label='train_acc')
    plt.plot(error['val_accuracy']*100, label='val_acc')
    plt.legend(loc='upper right')

    filename = 'model_performance.png'
    plt.savefig(filename)
    plt.show()
