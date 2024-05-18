# Import all required modules
import argparse
import os
import time
# Scikit-learn
from sklearn.metrics import classification_report,confusion_matrix
# Tensorflow
import tensorflow as tf
from classes import image_generator
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def main(image_dir, batchsize, model_path):

    # Input image parameters
    image_size = 128

    if os.path.exists(model_path):
        start_time = time.time()
        model = load_model(model_path)
        # load test data
        test_gen = image_generator(image_size, batchsize, image_dir, False)

        with tf.device('/CPU:0'):
            start_time = time.time()
            result = model.predict(test_gen, batch_size=batchsize, verbose=0)
            y_pred = [1 if pred >= 0.5 else 0 for pred in result]
            #y_pred = np.argmax(result, axis = 1)
            y_true = test_gen.labels
            loss,acc = model.evaluate(test_gen, batch_size=batchsize, verbose=0)
            labels =['Cat','Dog']
            end_time = time.time()
            print("Time taken for evaluation = ", end_time-start_time)
        
        print(classification_report(y_true, y_pred, target_names=labels))
        print('The loss of the model for test data is:', loss)  
        print('The accuracy of the model for test data is:', acc*100)  
        # Plot confusion matrix
        confusion_mtx = confusion_matrix(y_true, y_pred) 
        f,ax = plt.subplots(figsize = (8,4),dpi=200)
        sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, cmap="gist_yarg_r", linecolor="black", fmt='.0f', ax=ax, cbar=False, xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label",fontsize=10)
        plt.ylabel("True Label",fontsize=10)
        plt.title("Confusion Matrix",fontsize=13)
        plt.show()  
    else:
        print('No trained Model is yet avaliable')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='folder where images are stored for testing')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--model_path', type=str, help='directory where trained model are saved')
    args = parser.parse_args()
        
    main(args.image_dir, args.batch_size, args.model_path)
