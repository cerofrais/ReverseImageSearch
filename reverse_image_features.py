import os
import keras
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import random
import time
import numpy as np
# from deployment import graph



if not os.path.exists('data/pre_trained_resnet50.h5'):
# model = keras.applications.vgg16(weights='imagenet',include_top=True)
    model = keras.applications.ResNet50(weights='imagenet',include_top=True)
#model.summary()
    model.save('data/pre_trained_resnet50.h5')
else:
    model = load_model('data/pre_trained_resnet50.h5')

graph = tf.get_default_graph()

def load_image(path):
    img = image.load_img(path,target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return img,x

def feature_extract(model=model):
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
    #feat_extractor.summary()
    return feat_extractor


def get_features(img_path):
    img, x = load_image(img_path)
    #print("shape of x: ", x.shape)
    model_1 = feature_extract()
    feats = model_1.predict(x)
    # print(y.shape)
    return feats

def encode_folder(folder_path,filepath):
    for each_file in sorted(os.listdir(folder_path)) :
        with open(filepath,'a') as f:
            f.write(str(each_file)+','+ ",".join([str(x) for x in get_features(os.path.join(folder_path,each_file)).reshape((2048,))])+'\n')
            print(each_file)

# if __name__ == '__main__':
    # get_features('data/045.computer-keyboard/045_0003.jpg')
    # encode_folder('/data/visualsearch/unclean_data/caltech_256/all_imgs','data/encode.csv')


