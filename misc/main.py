
from __init__ import *

from keras_applications import resnet50
img_width =224
img_height = 224

def read_img_file(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def resize_img_to_array(img, img_shape=(224, 224)):
    img_array = np.array(
        img.resize(
            img_shape
        )
    )
    return img_array


def make_resnet_model(input_shape=[224,224,3]):
    model = ResNet50(input_shape=input_shape,
                     weights='imagenet',
                     include_top=False)
    for layer in model.layers:
        layer.trainable = False
    return model


def get_conv_feat(f, model):

    img = read_img_file(f)
    np_img = resize_img_to_array(img, img_shape=(img_width, img_height))
    X = preprocess_input(np.expand_dims(np_img, axis=0).astype(np.float))
    X_conv = model.predict(X)
    X_conv_2d = X_conv[0].flatten()

    return X_conv_2d

def test(path):
    x=read_img_file(path)
    np_img=resize_img_to_array(x)
    #print(x.shape)
    model = make_resnet_model(input_shape=[224, 224, 3])
    X = preprocess_input(np.expand_dims(np_img, axis=0).astype(np.float))
    X_conv = model.predict(X)
    print(X_conv.shape)
    pass


if __name__ == '__main__':
    #print(os.listdir(datapath))
    #x=get_conv_feat('data/045.computer-keyboard/045_0001.jpg',make_resnet_conv())
    #print(x.shape)
    make_resnet_model().summary()