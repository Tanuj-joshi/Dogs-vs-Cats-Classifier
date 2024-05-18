# Import all required modules
import argparse
import os
import time
# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import csv


def main(image_dir, batchsize, model_path):

    if os.path.exists(model_path):
        # Input image parameters
        image_size = 128
        # loading into dataframe
        filenames = os.listdir(image_dir)
        test_data = pd.DataFrame({"filename": filenames})
        test_data['label'] = 'unknown'
        # Create data genenerator for test data
        datagen = ImageDataGenerator(rescale=1./255)
        test_idg =  datagen.flow_from_dataframe(test_data, 
                                                image_dir, 
                                                x_col= "filename",
                                                y_col = "label",
                                                batch_size = batchsize,
                                                target_size=(image_size, image_size), 
                                                shuffle = False)
        model = load_model(model_path)
        with tf.device('/CPU:0'):
        # generate prediction for test images
            test_predict = model.predict(test_idg, batch_size=batchsize, verbose=0)
            y_test_pred = [1 if pred >= 0.5 else 0 for pred in test_predict]
            #y_test_pred = np.argmax(test_predict, axis = 1)
            test_data['label'] = y_test_pred
            # mapping
            label_mapping = {0: 'cat', 1: 'dog'}
            test_data['label'] = test_data['label'].map(label_mapping)
            print(test_data)
            # csv file output
            test_data.to_csv('test_results.csv', index=False)
    else:
        print('No trained Model is yet avaliable')
    
    # Visualize prediction results for few sample images
    fig, axes = plt.subplots(1, 20, figsize=(30, 4))
    for idx in range(20):
        image_path = os.path.join(image_dir, test_data.iloc[idx]['filename'])
        image = Image.open(image_path)
        axes[idx].imshow(image)
        axes[idx].set_title(test_data.iloc[idx]['label'])
        axes[idx].axis('off')
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='folder where images are stored for Testing')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--model_path', type=str, help='directory where trained model are saved')

    args = parser.parse_args()
        

    main(args.image_dir, args.batch_size, args.model_path)

