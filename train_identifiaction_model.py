import model_utils
import keras
from keras.utils import multi_gpu_model
import utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

weight_load_path = "weights_7.h5"
weight_save_path = "weights.h5"
optimizer = keras.optimizers.Adam
lr = 0.1
epochs = 5
batch_size = 16
check_point_idx = batch_size * 100
image_shape = [224, 224, 3]

if __name__ == "__main__":
    # Config GPU options
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #set_session(sess)

    # Get data
    model = model_utils.identification_model(image_shape, 2, weight_load_path)
    #parallel_model = multi_gpu_model(model, gpus=3)
    model.compile(optimizer(lr=lr), loss=model_utils.contrastive_loss, metrics=["accuracy"])
    # Begin training
    current_idx, current_epochs = 0, 0
    # Get the whole data
    print("Loading data:")
    data, annotation, view_predictions , names = utils.load_all_data()
    annotation.set_index("Image", inplace=True)
    train_portion = 0.7
    num_train = int(len(data)*train_portion)
    train_data = data[:num_train]
    train_annotation = annotation[:num_train]
    train_view_predictions = view_predictions[:num_train]
    print("Done!")
    while(current_epochs < epochs):
        st = time.time()
        current_batch_size = batch_size
        # Fetch the next batch
        print("%d/%d" %(current_idx+batch_size, num_train), end=": ")
        batch, random_batch, ground_truth, batch_view, random_batch_view = \
            utils.get_next_batch(train_data, train_annotation, train_view_predictions,
                                 names, current_idx, batch_size)
        current_batch_size = len(batch)
        # Feed the batches
        loss, acc = model.train_on_batch(
            [batch, random_batch, batch_view, random_batch_view], ground_truth,
        class_weight={0:1, 1:100})
        t = time.time() - st
        print("batch loss: %f, acc: %f" %(loss, acc), end=", ")
        print("time:", t)
        current_idx += current_batch_size
        # If we iterate over all samples, increment current_epochs
        if current_idx % check_point_idx == 0:
            model.save_weights("weights_%d_%d.h5" %(current_idx, current_epochs))
        if current_idx >= num_train:
            current_idx = 0
            current_epochs += 1
            model.save_weights("weights_%d.h5" %(current_epochs))
    model.save_weights(weight_save_path)