from keras import Model, Input, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, \
Lambda, Concatenate, Flatten, Dropout, Activation, BatchNormalization
from keras.applications.resnet50 import ResNet50
import keras
import tensorflow as tf

# Define constant
NUM_NEGATIVE_SAMPLES = 5

#Define a loss function where target matrix might have missing values
#T.shape = Y.shape = [batch_size, NUM_NEGATIVE_SAMPLES+1]
# def nce_crossentropy(T, Y):
#     transposed_T = tf.transpose(T)
#     transposed_Y = tf.transpose(Y)
#     positive_T = tf.gather(transposed_T, 0)
#     positive_Y = tf.gather(transposed_Y, 0)
#     negative_T = tf.gather(transposed_T, [i+1 for i in range(NUM_NEGATIVE_SAMPLES)])
#     negative_Y = tf.gather(transposed_Y, [i+1 for i in range(NUM_NEGATIVE_SAMPLES)])
#
#     positive_loss = -(positive_T*tf.log(positive_Y) + (1-positive_T)*tf.log(1-positive_Y))
#     negative_loss = t*tf.log(y) + (1-t)*tf.log(1-y)
#     return tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)

def identification_model(image_shape, model_num):
    with tf.device("/cpu:0"):
        # Define model
        img = Input(shape=image_shape)
        img1 = Input(shape=image_shape)
        img2 = Input(shape=image_shape)
        resnet = ResNet50(include_top=False)
        baseModel = Model(inputs=[img],
                          outputs=GlobalAveragePooling2D()(resnet(img)))
        embedded_img1 = baseModel(img1)
        embedded_img2 = baseModel(img2)
        L1_layer = Lambda(lambda embeded_pair: tf.abs(embeded_pair[0] - embeded_pair[1]))
        L1_distance = L1_layer([embedded_img1, embedded_img2])
        if model_num == 1:
            view_model = view_classification_model(False, "view_model_weights.h5")
            merged = Concatenate([L1_distance, view_model])
            fully1 = Dense(100, activation="relu", kernel_initializer="he_uniform")(merged)
            out = Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")(fully1)
        elif model_num == 2:
            out = Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")(L1_distance)
        else:
            return None
        model = Model(inputs=[img1, img2], outputs=out)

    return model

def view_classification_model(trainable=False, weight_path=None):
    with tf.device("/cpu:0"):
        model = Sequential()
        model.add(Flatten(input_shape=[256, 256]))
        model.add(Dense(512, kernel_initializer="he_uniform",
                        trainable=trainable, kernel_regularizer=keras.regularizers.l2()))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, kernel_initializer="he_uniform",
                        trainable=trainable, kernel_regularizer=keras.regularizers.l2()))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation="sigmoid", trainable=trainable,
                        kernel_regularizer=keras.regularizers.l2()))
        if weight_path is not None:
            model.load_weights(weight_path)
    return model