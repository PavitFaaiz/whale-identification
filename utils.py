import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import pickle
from keras.preprocessing import image
import cv2
import tensorflow as tf

def load_all_data():
    with tf.device("/cpu:0"):
        if not os.path.exists(os.getcwd()+"\\data.p"):
            image_path = "resized\\"
            annotate_path = "train.csv"

            #Process samples
            filenames = os.listdir(image_path)
            data = np.zeros([len(filenames), 224, 224 ,3], dtype=np.float16)
            for idx, f in enumerate(filenames):
                img = Image.open(image_path + f)
                x = image.img_to_array(img)
                if len(x.shape) == 2:
                    x = np.expand_dims(x, axis=3)
                data[idx] = x
                img.close()

            ##Process annotations
            df = pd.read_csv(annotate_path, index_col=False)
            pickle.dump((data, df), open("data.p", "wb")) #Save the data with pickle to be easily accessed later
        else:
            data, df = pickle.load(open("data.p", "rb"))
    return data, df

def load_view_data():
    with tf.device("/cpu:0"):
        if not os.path.exists("view_data.p"):
            #Process samples
            front_filenames = list(np.array(os.listdir("front")))
            back_filenames = list(np.array(os.listdir("back")))
            num_front = len(front_filenames)
            num_back = len(back_filenames)
            num_samples = num_front + num_back
            data = np.zeros([num_samples, 256, 256], dtype=np.float16)
            for idx, f in enumerate(front_filenames):
                x = view_classification_preprocess("front\\" + f)
                data[idx] = x
                if idx%100==0:
                    print(idx)

            for idx, f in enumerate(back_filenames):
                x = view_classification_preprocess("back\\" + f)
                data[idx + num_front] = x
                if idx%100==0:
                    print(idx)

            ##Process labels:
            # Front <= 0
            # Back <= 1
            labels = np.append(np.zeros(num_front), np.ones(num_back))
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            #Process training, testing sets
            train_portion = 0.7
            num_train = int(num_samples * train_portion)
            train_samples = data[:num_train]
            test_samples = data[num_train:]
            train_labels = labels[:num_train]
            test_labels = labels[num_train:]

            # Save the data with pickle to be easily accessed later
            pickle.dump(((train_samples, train_labels), (test_samples, test_labels)),
                        open("view_data.p", "wb"))
        else:
            (train_samples, train_labels), (test_samples, test_labels) = pickle.load(open("view_data.p", "rb"))
    return (train_samples, train_labels), (test_samples, test_labels)

def show_image(filename, winname="img"):
    img = cv2.imread(filename)
    cv2.imshow(winname, img)

def save_image(filename, path):
    img = cv2.imread(filename)
    cv2.imwrite(path, img)

def resize_all_images():
    filenames = os.listdir("train")
    for idx, name in enumerate(filenames):
        save_name = name[0:len(name)-4]+".png"
        img = preprocess_image("train\\"+name)
        img.save("resized\\" + save_name)
        img.close()

def preprocess_image(img_path, image_width=224):
    #Load image
    img = Image.open(img_path)
    width, height = img.size
    #Modify the image shape
    maxx = max(width, height)
    factor = maxx/image_width
    size = int(np.floor(img.size[0]/factor)), int(np.floor(img.size[1]/factor))
    img = img.resize(size)
    delta_w = np.abs(size[0]-image_width)
    delta_h = np.abs(size[1]-image_width)
    pad = tuple(np.array((np.ceil(delta_w/2), np.ceil(delta_h/2),
                          np.floor(delta_w/2), np.floor(delta_h/2)), dtype=np.int))
    img = ImageOps.expand(img, pad) #Pad the smaller dimension to make it square
    return img

def view_classification_preprocess(img_path):
    # Load an image from the front view images and convert to HSV color space
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get each channel flatten
    S_channel = img[:, :, 1].ravel()
    V_channel = img[:, :, 2].ravel()
    # Calculate 2d histogram and sum
    hist = np.histogram2d(S_channel, V_channel, bins=[256, 256])[0]
    # Normalize the histogram
    hist = hist / np.max(hist)
    return hist

def get_next_batch(data, annotation, current_index, batch_size):
    with tf.device("/cpu:0"):
        batch_size = min(batch_size, len(data) - current_index)
        filenames = os.listdir("resized")
        #Process annotations
        image_column = annotation["Image"]
        annotation["Image"] = image_column.str.\
            replace(".jpg", ".png") #The preprocessed files are in .png format
        ## Get next batch
        batch = data[current_index:current_index + batch_size, :]

        ## Get the other distinct random batch
        indices = np.arange(len(data))
        #Remove the current batch from the random sampled ones so no duplicates
        indices = np.delete(indices, np.arange(current_index, current_index+32,))
        np.random.shuffle(indices)
        random_batch = data[indices[:batch_size]]

        ## Process ground truth values
        ground_truth = np.zeros([batch_size])
        for i in range(batch_size):
            id = annotation["Id"][current_index+i]
            random_sample_id = annotation["Id"][indices[i]]
            if id == random_sample_id:
                ground_truth[i] = 1
    print("fetched, ", end="")
    return batch, random_batch, ground_truth

load_view_data()