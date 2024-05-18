# Import all required modules
import argparse
import os
import time
from os import makedirs
# Scikit-learn
from sklearn.model_selection import train_test_split
# Tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,TensorBoard,ModelCheckpoint
from classes import extract_data, visualize_data, train_val_split, model_design, image_generator,generate_plots
from tensorflow.keras.models import load_model



def main(image_dir, batchsize, epoch, patience, model_path):

    # Input image parameters
    image_size = 128
    image_channel = 3

    # If training is carried out
    '''
    # Extract Tmage Data from zip files
    input_train_dir = './dogs-vs-cats/train.zip'
    input_test_dir = "./dogs-vs-cats/test1.zip"
    output_dir = "./dogs-vs-cats/"
    if not os.path.exists(input_train_dir) or not os.path.exists(input_test_dir):
        print('Data is not available')
    else:
        extract_data(input_train_dir, input_test_dir, output_dir)

    # Visualize few images
    visualize_data('dogs-vs-cats/train/*.jpg')

    # Split the Dataset into train and val set and for each category 
    train_val_split('dogs-vs-cats/train/', 'Dataset/', 0.2)
    '''
    # path where we want to save the trained model
    makedirs(model_path, exist_ok=True)
    # model logs
    log_dir = 'logs'
    makedirs(log_dir, exist_ok=True)
    # Applying image data gernerator to train and validation data
    train_gen = image_generator(image_size, batchsize, str(image_dir)+'train', True)
    val_gen = image_generator(image_size, batchsize, str(image_dir)+'val', False)
    # Defining Callbacks
    # 1): Reduce learning rate when a metric has stopped improving.
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                factor=0.5,
                                                min_lr=0.00001,
                                                verbose=1)
    # 2): Stop training when a monitored metric has stopped improving.
    early_stoping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    # 3): To generate plots for accuracy and losses over time
    tensorboard = TensorBoard(log_dir=log_dir +"/{}".format(time.time()))
    # 4): To save trained model at specific intervals and conditions
    model_checkpoint = ModelCheckpoint(filepath=model_path, 
                                        monitor='val_loss',
                                        save_weights_only=False,
                                        verbose=0,
                                        save_best_only=False,
                                        mode='auto', 
                                        save_freq='epoch')

    callbacks = [early_stoping, learning_rate_reduction, tensorboard, model_checkpoint]

    # Create and load the CNN model 
    model = model_design(image_size, image_channel)
    model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # Start Training/Fitting the model
    with tf.device('/GPU:0'):
        start_time = time.time()
        hist = model.fit(train_gen,
                            validation_data=val_gen, 
                            callbacks=callbacks,
                            epochs=epoch,
                            steps_per_epoch = len(train_gen),
                            validation_steps = len(val_gen),
                            )
        end_time = time.time() 
        # generate plots for accuracy and Loss with epochs
        generate_plots(hist)
        print("Time taken for training = ", end_time-start_time)
    #model.save('alternative_path/')       



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='folder where images are stored for training and validation')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--epoch', type=int, help='total training epochs')
    parser.add_argument('--patience', type=int, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--model_path', type=str, help='directory where trained model are saved')
    args = parser.parse_args()
        
    if args.epoch and args.batch_size != None:
        main(args.image_dir, args.batch_size, args.epoch, args.patience, args.model_path)

