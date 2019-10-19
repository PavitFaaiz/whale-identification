from keras import Model, Input, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, \
Lambda, concatenate, Flatten, Dropout, Activation, BatchNormalization
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


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def identification_model(image_shape, model_num, weight_path=None):
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
        L1_layer = Lambda(lambda embeded_pair:
                          tf.reduce_sum(tf.pow(embeded_pair[0] - embeded_pair[1],2.0), axis=1, keepdims=True))
        if model_num == 1:
            L1_distance = L1_layer([embedded_img1, embedded_img2])
        elif model_num == 2:
            # Reduce dimension and scale values to [0, 1] (same scale as view predictions)
            embedded_img = Input(shape=[2048])
            fc1_layer = Model(inputs=embedded_img,
                outputs=Dropout(0.5)(Activation("sigmoid")(BatchNormalization()(Dense(128)(embedded_img)))))
            embed_128_1 = fc1_layer(embedded_img1)
            embed_128_2 = fc1_layer(embedded_img2)

            # Concatenate the embeddings with their view predictions
            view1 = Input(shape=[1])
            view2 = Input(shape=[1])
            merged1 = concatenate([embed_128_1, view1])
            merged2 = concatenate([embed_128_2, view2])

            # Reduce dimension to 32-dimension vectors
            merged = Input(shape=[129])
            fc2_layer = Model(inputs=merged,
                outputs=Dropout(0.5)(Activation("sigmoid")(BatchNormalization()(Dense(32)(merged)))))
            embed_32_1 = fc2_layer(merged1)
            embed_32_2 = fc2_layer(merged2)
            norm_layer = Lambda(lambda tensor: tensor/tf.norm(tensor))
            normed1 = norm_layer(embed_32_1)
            normed2 = norm_layer(embed_32_2)
            L1_distance = L1_layer([normed1, normed2])
        else:
            return None
        #out = Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")(L1_distance)
        model = Model(inputs=[img1, img2, view1, view2], outputs=L1_distance)
        if not weight_path is None:
            model.load_weights(weight_path)
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