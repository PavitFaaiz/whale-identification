import model_utils
import keras
from keras.utils import multi_gpu_model
import utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time

weight_load_path = None
weight_save_path = "weights.h5"
optimizer = keras.optimizers.Adam
lr = 0.001
epochs = 5
batch_size = 16
check_point_idx = batch_size * 10
image_shape = [224, 224, 3]

if __name__ == "__main__":
    # Config GPU options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # Get data
    model = model_utils.get_model(image_shape)
    #parallel_model = multi_gpu_model(model, gpus=3)
    if weight_load_path is not None:
        model.load_weights(weight_load_path)
    # Begin training
    current_idx, current_epochs = 0, 0
    # Get the whole data
    print("Loading data:")
    data, annotation = utils.load_data("")
    print("Done!")
    while(current_epochs < epochs):
        st = time.time()
        current_batch_size = batch_size
        # Fetch the next batch
        print("%d/%d" %(current_idx, len(data)), end=": ")
        batch, random_batch, ground_truth = utils.get_next_batch(data, annotation, current_idx, batch_size)
        current_batch_size = len(batch)
        # Feed the batches
        loss = model.train_on_batch([batch, random_batch], ground_truth)
        t = time.time() - st
        print("batch loss:", loss, end=", ")
        print("time:", t)
        current_idx += current_batch_size
        # If we iterate over all samples, increment current_epochs
        if current_idx % check_point_idx == 0:
            model.save_weights("weights_%d_%d.h5" %(current_idx, current_epochs))
        if current_idx >= len(data):
            current_idx = 0
            current_epochs += 1
    model.save_weights(weight_save_path)