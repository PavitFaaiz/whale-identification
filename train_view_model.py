import model_utils
import keras
import numpy as np
import matplotlib.pyplot as plt
import utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

os.environ["tf_cpp_min_log_level"] = "3"
weight_load_path = None
weight_save_path = "view_model_weights_30.h5"
optimizer = keras.optimizers.Adam
lr = 0.001
epochs = 10
batch_size = 32
check_point_idx = batch_size * 10
image_shape = [224, 224, 3]

if __name__ == "__main__":
    # Config GPU options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # Get data
    model = model_utils.view_classification_model(True, "view_model_weights_20.h5")
    #parallel_model = multi_gpu_model(model, gpus=3)
    model.compile(optimizer=optimizer(lr=lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    # Begin training
    # Get the whole data
    print("Loading data:")
    (train_samples, train_labels), (test_samples, test_labels) = utils.load_view_data()
    print("Done!")
    model.fit(train_samples, train_labels, epochs=epochs,
              validation_data=(test_samples, test_labels))
    model.save_weights("view_model_weights_30.h5")
    model.fit(train_samples, train_labels, epochs=epochs,
              validation_data=(test_samples, test_labels))
    model.save_weights("view_model_weights_40.h5")
    model.fit(train_samples, train_labels, epochs=epochs,
              validation_data=(test_samples, test_labels))
    model.save_weights("view_model_weights_50.h5")