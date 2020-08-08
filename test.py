
from __init__ import *
from reverse_image_features import *


df = pd.read_csv('data/encode.csv', header= None) # 2569, 2049
#print(df.head())
print(df.shape)

Y = df[0]

df_feats = df[[_ for _ in df.columns if _ != 0]].values.astype(np.float)
print(df_feats.shape)


def knnmodel():
    knn = NearestNeighbors(n_neighbors=5, n_jobs=8, algorithm='ball_tree')
    knn.fit(df_feats)
    joblib.dump(knn, 'data/knn.pkl')


if not os.path.exists('data/knn.pkl'):
    knnmodel()

knn =  joblib.load('data/knn.pkl')

def get_neighbors(knn,imagepath):
    test_img_features = get_features(imagepath)
    neighbors = knn.kneighbors(test_img_features, return_distance=True)
    return neighbors

def get_names(neighbors):
    result = []
    for each in neighbors[1].flatten():
        result.append(Y.tolist()[each])
        shutil.copy(pjoin('data/all_imgs', result[-1]),'static/output')
    print(result)
    return result

#test_img_features = get_features('data/045.computer-keyboard/045_0003.jpg')
#get_names(get_neighbors(knn,'data/127.laptop-101/127_0008.jpg'))