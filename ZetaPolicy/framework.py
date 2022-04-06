import datetime
import logging
import os
import pickle
import sys
from os import path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ZetaPolicy.constants import DATA_ROOT


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def get_dataset(filename='signal-dataset.pkl'):
    if not path.exists(DATA_ROOT + filename):
        download(filename)

    with open(DATA_ROOT + filename, 'rb') as f:
        data = pickle.load(f)
        return data


def download(filename, base_url='https://s3-ap-southeast-1.amazonaws.com/usq.iothealth/iemocap/'):
    import urllib.request

    url = base_url + filename

    print('Beginning file download {}'.format(url))

    store_file = DATA_ROOT + filename
    urllib.request.urlretrieve(url, store_file)

    print("Downloaded and saved to file: {}".format(store_file))


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(
            f"There is no such a hdf5 file ({hdf5_name}). \nDownload from here: https://s3.ap-southeast-2.amazonaws.com/usq.iothealth.sidney/iemocap/{hdf5_name}")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning("Dataset in hdf5 file already exists. "
                                "recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error("Dataset in hdf5 file already exists. "
                              "if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def train(model, x, y, epochs, batch_size=4, log_base_dir='./logs'):
    print("Start Training")
    log_dir = log_base_dir + "/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=log_base_dir + "/" + model.name + '.h5',
            monitor='val_accuracy',
            save_best_only='True',
            verbose=1,
            mode='max'
        ), tensorboard_callback]

    history = model.fit(x, y,
                        batch_size=batch_size, epochs=epochs,
                        validation_split=0.2,
                        verbose=True,
                        callbacks=callback_list)
    return history, model
