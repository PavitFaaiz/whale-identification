from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import pickle
from keras.preprocessing import image
from PIL import ImageOps, Image
import numpy as np
import os
from keras.layers import GlobalAveragePooling2D
from keras import Model, Input
#define model
X = Input(shape=[224, 224, 3])
resnet50 = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=X, outputs=GlobalAveragePooling2D()(resnet50(X)))
filenames = os.listdir("train")
idx = 0
names = [None for _ in filenames]
embedded = np.zeros([len(filenames), 2048])
while idx < len(filenames):
    start = idx
    batch_size = 100
    if idx + 100 > len(filenames):
        batch_size = len(filenames) - idx
    batch = np.zeros([batch_size, 224, 224, 3])
    for i in range(batch_size): #Read 100 images at a time
        name = filenames[idx]
        names[idx] = name
        img_path = 'train\\' + name
        img = Image.open(img_path)
        #Modify the image shape
        width, height = img.size
        maxx = max(width, height)
        factor = maxx/224
        size = int(np.floor(img.size[0]/factor)), int(np.floor(img.size[1]/factor))
        img = img.resize(size)
        smaller_dim = int(np.argmin(size))
        delta_w = np.abs(size[0]-224)
        delta_h = np.abs(size[1]-224)
        padd = tuple(np.array((np.ceil(delta_w/2), np.ceil(delta_h/2),
                np.floor(delta_w/2), np.floor(delta_h/2)), dtype=np.int))
        img = ImageOps.expand(img, padd)
        x = image.img_to_array(img)
        img.close()
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=3)
        batch[i] = x
        idx += 1
    print(idx)
    batch = preprocess_input(batch)
    preds = model.predict(batch)
    embedded[start:idx] = preds

pickle.dump((names, embedded), open("embedded_images.p", "wb"))
