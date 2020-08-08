import cv2
import shutil
import time
from os.path import join as pjoin
from os import listdir
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pyzbar.pyzbar as pyzbar

import joblib
from reverse_image_features import *
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.storage.blob import BlockBlobService

def train(csvpath):
    df_feats, Y = csv_to_knn(csvpath)
    y_dump = csvpath.split("/")[-1][:-4]
    joblib.dump(Y, 'data/ignite_demo/Y_' + y_dump + '.pkl')
    path = 'data/ignite_demo/knn_' + y_dump + '.pkl'
    knnmodel(df_feats, path)
    
def csv_to_knn(csvpath):
    df = pd.read_csv(csvpath, header=None)  # 2569, 2049
    # print(df.head())
    Y = df[0]
    df_feats = df[[_ for _ in df.columns if _ != 0]].values.astype(np.float)
    # print(df_feats.shape)
    return df_feats, Y    

def make_csvs():
    """
    Takes a list of folders and generates csv files with image name and Resnet50 features.
    :return: Nothing
    """
    encode_folder('data/ignite_demo/all_images', 'data/ignite_demo/all_images.csv')
    encode_folder('data/ignite_demo/DisposableGloves', 'data/ignite_demo/DisposableGloves.csv')
    encode_folder('data/ignite_demo/ScrewDriver', 'data/ignite_demo/ScrewDriver.csv')

# make_csvs()    

def knnmodel(df_feats, path):
    """
    Training a Knn model
    :param df_feats:
    :param path:
    :return:
    """
    knn = NearestNeighbors(n_neighbors=10, n_jobs=-2, algorithm='ball_tree')
    knn.fit(df_feats)
    joblib.dump(knn, path)
 
    
def get_neighbors(knn, imagepath):
    test_img_features = get_features(imagepath)
    neighbors = knn.kneighbors(test_img_features, return_distance=True)
    return neighbors    

def get_names(Y, neighbors,localbool):
    result = []
    for each in neighbors[1].flatten():
        result.append(Y.tolist()[each])
        # this is needed only for demo via chrome:
        if localbool:
            shutil.copy(pjoin('data/ignite_demo/all_images', result[-1]), 'static/output')
    # print(result)
    return result


if not os.path.exists('data/ignite_demo/knn_all_images.pkl'):
    print("Training")
    train('data/ignite_demo/all_images.csv')
knn = joblib.load('data/ignite_demo/knn_all_images.pkl')
Y = joblib.load('data/ignite_demo/Y_all_images.pkl')


def azure_detect_object(img_path):
    """
    detect gloves and screwdriver in given image
    :param img_path: Image path
    :return:
    """
    res = {}
    template_prediction_key = "54c71598e9434d5fa7853360c4a9e4ce"
    template_project_id = "54e7f828-d0c8-49d3-8802-9b402612b7c7"
    template_iteration_name = "Iteration2"
    template_prediction_endpoint = "https://southeastasia.api.cognitive.microsoft.com"
    predictor = CustomVisionPredictionClient(template_prediction_key, endpoint=template_prediction_endpoint)

    with open(img_path, "rb") as image_contents:
        results = predictor.detect_image(
            template_project_id,
            template_iteration_name,
            image_contents.read(),
            custom_headers={'Content-Type': 'application/octet-stream'})
        res = []
        for prediction in results.predictions:

            res.append(
                (prediction.tag_name,
                 prediction.probability,
                 prediction.bounding_box.left,
                 prediction.bounding_box.top,
                 prediction.bounding_box.width,
                 prediction.bounding_box.height))
    return res


def draw_(imgpath, res, crops_path):
    """
    draw and crop the image based on Azure detection
    :param imgpath: imagepath
    :param res: detection coords
    :return:
    """
    ims = []
    raw_im = cv2.imread(imgpath)
    for i, each in enumerate(res):
        tag, pa, x, y, w, h = list(each)
        if pa < 0.43:
            continue
        img = cv2.imread(imgpath)
        wi, he = img.shape[0], img.shape[1]
        crop = img[int(y * wi): int((y + h) * wi), int(x * he):int((x + w) * he)]
        croppath = pjoin(crops_path,'crop'+ str(time.time()).replace(".","_") + "_" + str(i) + '.jpg')
        cv2.imwrite(croppath, crop)
        # print((int(x * he), int(y * wi)), (int((x + w) * he), int((y + h) * wi)))
        cv2.rectangle(img, (int(x * he), int(y * wi)), (int((x + w) * he), int((y + h) * wi)), (255, 0, 0), 2)
        cv2.rectangle(raw_im, (int(x * he), int(y * wi)), (int((x + w) * he), int((y + h) * wi)), (255, 0, 0), 2)
        # cv2.imshow('res' + str(i), img)
        detect_path = "/".join(crops_path.split("/")[:-1])
        decpath = pjoin(detect_path,'detect/dec' + str(i) + '.jpg')
        # print(tag, pa)
        # print(img.shape)
        cv2.imwrite(decpath, img)
        ims.append([tag, pa, decpath])
    alldetect_path = 'static/input/in_dec.jpg'
    cv2.imwrite(alldetect_path, raw_im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return ims
  
  
def driver_crop_image(img_path,crops_path,filename,localbool=0,out_path=''):
    res = azure_detect_object(img_path)
    draw_(img_path, res,crops_path) 
    results = []
    if len(os.listdir(crops_path))==0:
        if localbool :
            res = get_names(Y, get_neighbors(knn, img_path),localbool)
            results = [{'crop_img': 'input/' + filename, 'out_img': ['output/' + e for e in res]}]
        else:
            res = get_names(Y, get_neighbors(knn, img_path), localbool)
            results = [{'crop_img': 'input/' + filename, 'out_img': res}]
    else :
        for each in os.listdir(crops_path):
            result = {}
            result['crop_img'] = 'crops/' + each
            res = get_names(Y, get_neighbors(knn, pjoin(crops_path,each)),localbool)
            result['out_img'] = res #['output/' + e for e in res]
            results.append(result)      
    return results


def decode(img_path):
    image = cv2.imread(img_path)
    decodedObjects = pyzbar.decode(image)
    return decodedObjects


# driver_class_whole_image('data/ignite_demo/ScrewDriver/screwdriver10.jpg')
# driver_class_whole_image('data/ignite_demo/DisposableGloves/gloves7.jpg')
# driver_crop_image('data/ignite_demo/ScrewDriver/screwdriver3.jpg')
