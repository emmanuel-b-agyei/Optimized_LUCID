#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized LUCID Setup Script
Author: Extended from the original by Roberto Doriguzzi-Corin
License: Apache License 2.0 (Original), Extensions under MIT License

This script contains:
- Optimized CNN model with SeparableConv2D
- Model compilation
- TFLite conversion with float16 quantization
- Threaded live loop for real-time prediction
- Config setup and reproducibility controls
- And multiprocessing coded in lucid_dataset.py
"""

import time
start_time = time.time()
import tensorflow as tf
import numpy as np
import random as rn
import os
import csv
import pprint
from util_functions import *
import datetime
import threading
import tensorflow.keras.backend as K
import threading
import optuna
from optuna.trial import FixedTrial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from lucid_dataset_parser import *
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from imblearn.over_sampling import SMOTE 


os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True 

'''
import datetime
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='1,6000')
'''

PATIENCE = 10
DEFAULT_EPOCHS = 1000
OUTPUT_FOLDER = "./output/"
VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
hyperparamters = {
    "learning_rate": [0.1,0.01],
    "batch_size": [1024,2048],
    "kernels": [32,64],
    "regularization" : [None,'l2'],
    "dropout" : [0.2,0.3]
}


# Optimized model with SeparableConv2D
def Conv2DModel(model_name,input_shape,kernel_col, kernels=64,kernel_rows=3,learning_rate=0.01,regularization='l2',dropout=0.3, verbose=True):
    K.clear_session()
    model = Sequential(name=model_name)
    model.add(SeparableConv2D(kernels, (kernel_rows,kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularization))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid', name='fc1'))
    if verbose: print(model.summary())
    compileModel(model, learning_rate)
    return model

def compileModel(model,lr):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy']) 

# Live Prediction Thread Function (H5 and TFLite)
def threaded_live_loop(
    cap, model, mins, maxs, max_flow_len, time_window,
    dataset_type, labels, predict_writer, predict_file,
    model_name_string, is_tflite=False
):
    if is_tflite:
        interpreter, input_details, output_details = model

    while True:
        samples = process_live_traffic(cap, dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)

        if len(samples) > 0:
            X, Y_true, _ = dataset_to_list_of_fragments(samples)
            X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len)).astype(np.float32)
            X = np.expand_dims(X, axis=-1) 
            Y_true = np.array(Y_true) if labels is not None else None

            pt0 = time.time()

            if is_tflite:
                Y_pred = []
                for sample in X:
                    input_tensor = np.expand_dims(sample, axis=0)  
                    interpreter.set_tensor(input_details[0]['index'], input_tensor)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]['index'])
                    pred = 1 if output[0][0] > 0.5 else 0
                    Y_pred.append(pred)
                Y_pred = np.array(Y_pred)
            else:
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)

            pt1 = time.time()
            prediction_time = pt1 - pt0

            [packets] = count_packets_in_dataset([X])
            report_results(Y_true, Y_pred, packets, model_name_string, "LIVE", prediction_time, predict_writer, predict_file)

        time.sleep(0.5)

# TFLite Conversion
def convert_to_tflite(model_path, quantize=True):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    
    tflite_model_path = model_path.replace(".h5", "_quantized.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] Saved quantized TFLite model to: {tflite_model_path}")


# MAIN
def main(argv):
    help_string = 'Usage: python3 lucid_cnn.py --train <dataset_folder> -e <epocs>'
    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')
    parser.add_argument('-cv', '--cross_validation', default=0, type=int,
                        help='Number of folds for cross-validation (default 0)')
    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')
    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')
    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')
    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic')
    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')
    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')
    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')
    args = parser.parse_args()

    
    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)
        
    # TRAINING
    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: 
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        for full_path in subfolders:
            full_path = full_path.replace("//", "/") 
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')
            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            
            X_train_smote = X_train.reshape((X_train.shape[0], -1)) 

            # SMOTE Oversampling
            print("Before SMOTE: ", np.bincount(Y_train))
            sm = SMOTE(random_state=SEED)
            X_train_res, Y_train_res = sm.fit_resample(X_train_smote, Y_train)
            print("After SMOTE: ", np.bincount(Y_train_res))

            X_train = X_train_res.reshape((-1,) + X_train.shape[1:])
            Y_train = Y_train_res

            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)
           
            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()
            print ("\nCurrent dataset folder: ", dataset_folder)
            model_name = dataset_name + "-LUCID"
            
            
            # Hyperparameter Optimization with Optuna
            def build_model(trial, input_shape, model_name, kernel_col):
                learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
                batch_size = trial.suggest_categorical("batch_size", [512, 1024])
                kernels = trial.suggest_categorical("kernels", [32, 64])
                dropout = trial.suggest_categorical("dropout", [0.2, 0.5])
                regularization = trial.suggest_categorical("regularization", [None, 'l2'])

                K.clear_session()
                model = Sequential(name=model_name)
                model.add(Conv2D(kernels, (3, kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularization))
                model.add(Dropout(dropout))
                model.add(Activation('relu'))
                model.add(GlobalMaxPooling2D())
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid', name='fc1'))
                compileModel(model, learning_rate)
                return model, batch_size

            def objective(trial):
                model, batch_size = build_model(trial, input_shape=X_train.shape[1:], model_name=model_name, kernel_col=X_train.shape[2])

                history = model.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=50,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10)]
                )

                val_accuracy = np.max(history.history['val_accuracy'])
                return val_accuracy

            print("[INFO] Starting Optuna hyperparameter search...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            best_params = study.best_params
            print("[INFO] Best hyperparameters found:", best_params)

            # Final Model Training
            final_model = Sequential(name=model_name)
            final_model.add(Conv2D(best_params['kernels'], (3, X_train.shape[2]), strides=(1, 1), input_shape=X_train.shape[1:], kernel_regularizer=best_params['regularization']))
            final_model.add(Dropout(best_params['dropout']))
            final_model.add(Activation('relu'))
            final_model.add(GlobalMaxPooling2D())
            final_model.add(Flatten())
            final_model.add(Dense(1, activation='sigmoid', name='fc1'))
            compileModel(final_model, best_params['learning_rate'])
            print(final_model.summary())

            best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
           
            final_model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=args.epochs,
                batch_size=best_params['batch_size'],
                callbacks=[
                    EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE),
                    ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', save_best_only=True, mode='max')
                    ]
            )

            final_model.save(best_model_filename + '.h5')
            convert_to_tflite(best_model_filename + '.h5', quantize=True)

            # Model Evaluation
            Y_pred_val = (final_model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            val_file = open(best_model_filename + '.csv', 'w', newline='')
            val_file.truncate(0)
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()
            val_file.flush()
            row = {
                'Model': model_name,
                'Samples': Y_pred_val.shape[0],
                'Accuracy': '{:05.4f}'.format(accuracy),
                'F1Score': '{:05.4f}'.format(f1_score_val),
                'Hyper-parameters': best_params,
                'Validation Set': glob.glob(dataset_folder + "/*" + '-val.hdf5')[0]
            }
            val_writer.writerow(row)
            val_file.close()
            print("Best parameters:", best_params)
            print("Best model path:", best_model_filename)
            print("F1 Score of the best model on the validation set:", f1_score_val)
    
    # PREDICTION MODE
    # Performs batch predictions on preprocessed datasets using a saved model (.h5 or .tflite).
    if args.predict is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()
        iterations = args.iterations
        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")
        
        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5") + glob.glob(args.predict + "/*.tflite")

        for model_path in model_list:
            model_filename = os.path.basename(model_path)
            filename_prefix = model_filename.split('-')[0] + '-' + model_filename.split('-')[1] + '-'
            model_name_string = model_filename.split(filename_prefix)[1].split('.')[0]

            is_tflite = model_path.endswith(".tflite")
            
            if is_tflite:
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            else:
                model = load_model(model_path)

            warm_up_file = dataset_filelist[0]
            if filename_prefix in os.path.basename(warm_up_file):
                X, Y = load_dataset(warm_up_file)
                if is_tflite:
                    input_data = np.expand_dims(X[0], axis=0).astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]['index'])
                else:
                    _ = model.predict(X[:1])

            for dataset_file in dataset_filelist:
                if filename_prefix not in os.path.basename(dataset_file):
                    continue

                X, Y = load_dataset(dataset_file)
                [packets] = count_packets_in_dataset([X])
                Y_true = Y
                Y_pred = []
                avg_time = 0

                for i in range(iterations):
                    pt0 = time.time()

                    if is_tflite:
                        for sample in X:
                            input_data = np.expand_dims(sample, axis=0).astype(np.float32)
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            Y_pred.append(1 if output_data[0][0] > 0.5 else 0)
                        Y_pred = np.array(Y_pred)
                    else:
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

                    pt1 = time.time()
                    avg_time += pt1 - pt0

                avg_time = avg_time / iterations
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, os.path.basename(dataset_file), avg_time, predict_writer, predict_file)
                predict_file.flush()
        predict_file.close()

    # LIVE MODE
    if args.predict_live is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        predict_filename = os.path.join(OUTPUT_FOLDER, f'predictions-{timestamp}.csv')
        predict_file = open(predict_filename, 'w', newline='')
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if args.model is None or (not args.model.endswith('.h5') and not args.model.endswith('.tflite')):
            print("No valid model specified! Must be a .h5 or .tflite file.")
            exit(-1)

        is_tflite = args.model.endswith('.tflite')

        model_filename = os.path.basename(args.model).strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()

        mins, maxs = static_min_max(time_window)
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        if is_tflite:
            interpreter = tf.lite.Interpreter(model_path=args.model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model_object = (interpreter, input_details, output_details)
        else:
            model_object = load_model(args.model)

        cap = pyshark.LiveCapture(interface=args.predict_live)

        thread = threading.Thread(
            target=threaded_live_loop,
            args=(
                cap,
                model_object,
                mins,
                maxs,
                max_flow_len,
                time_window,
                args.dataset_type,
                labels,
                predict_writer,
                predict_file,
                model_name_string,
                is_tflite  
            )
        )
        thread.daemon = True
        thread.start()

        print("[INFO] Live prediction thread started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[INFO] Stopping live capture...")
            predict_file.close()
            exit(0)

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer, file_obj):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])
    if Y_true is not None and len(Y_true.shape) > 0: 
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)
    
    # Printing the results to the terminal
    print("Model:", model_name)
    print("Time:", '{:04.3f}'.format(prediction_time))
    print("Packets:", packets)
    print("Samples:", Y_pred.shape[0])
    print("DDOS%:", ddos_rate)
    print("Accuracy:", '{:05.4f}'.format(accuracy) if Y_true is not None else "N/A")
    print("F1Score:", '{:05.4f}'.format(f1) if Y_true is not None else "N/A")
    print("TPR:", '{:05.4f}'.format(tpr) if Y_true is not None else "N/A")
    print("FPR:", '{:05.4f}'.format(fpr) if Y_true is not None else "N/A")
    print("TNR:", '{:05.4f}'.format(tnr) if Y_true is not None else "N/A")
    print("FNR:", '{:05.4f}'.format(fnr) if Y_true is not None else "N/A")
    print("Source:", data_source)
    file_obj.flush()


if __name__ == "__main__":
    main(sys.argv[1:])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
