
# Other functions taken from ignite_demo.py file and stored here
    




def rename_files(folder_path, name):
    """
    Rename filse in folder
    :param folder_path:
    :param name:
    :return:
    """
    for i, each in enumerate(listdir(folder_path)):
        src = pjoin(folder_path, each)
        dest = pjoin(folder_path, name + str(i) + '.' + each.split('.')[-1])
        shutil.move(src, dest)



def make_csvs():
    """
    Takes a list of folders and generates csv files with image name and Resnet50 features.
    :return: Nothing
    """
    encode_folder('data/ignite_demo/all_images', 'data/ignite_demo/all_images.csv')
    encode_folder('data/ignite_demo/DisposableGloves', 'data/ignite_demo/DisposableGloves.csv')
    encode_folder('data/ignite_demo/ScrewDriver', 'data/ignite_demo/ScrewDriver.csv')

# make_csvs()


def azure_classify(img_path):
    classification_key = "54c71598e9434d5fa7853360c4a9e4ce"
    project_id = "858cb625-b783-405d-acf0-f7b99d077e57"
    iteration_name = "Iteration1"
    res = {}
    classification_endpoint = "https://southeastasia.api.cognitive.microsoft.com"
    predictor = CustomVisionPredictionClient(classification_key, endpoint=classification_endpoint)

    with open(img_path, "rb") as image_contents:
        results = predictor.classify_image(
            project_id,
            iteration_name,
            image_contents.read(),
            custom_headers={'Content-Type': 'application/octet-stream'})

        for prediction in results.predictions:
            #print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
            res[prediction.tag_name] = prediction.probability * 100
    ans = max(res, key=res.get)
    prob = max(res)
    print(ans, prob)
    return ans, prob

def test_driver():
    for imgs in listdir('data/ignite_demo/testimage'):
        inp = pjoin('data/ignite_demo/testimage', imgs)
        print(inp)
        res = azure_detect_object(inp)
        draw_(inp, res)
    pass



#this is taken from ignite_demo_app.py

@app.route('/decoder_test', methods=["POST", "GET"])
def decoder_test(): 
    '''testing bar code samples'''  
    print("working")
    decodeobj_op = []
    bar_code_list = []
    barcode_samples = ['GRA-13G266','19811','923458','2DGP6','5LW89','GRA-10J217',
                       '5LW55','2AJK2','GRA-23NN72','2358','16953','MCM-59215A43',
                       'GRA-5KPJ1','923437','GRA-1YUF9','GRA-53KG69','MOT-0153-42C',
                       'AMAZ-45003','5176','AMZ-45005','MCM-5497a48','1765','100161',
                       '1492398','1491308','1491310','1412794','79201078','1387508',
                       '1387503','1387505','23670224','34860240','1387506']
    folder_path = r'C:\Users\ajayeswar.reddy\Downloads\bar_code_new'
    for file in os.listdir(folder_path):
        path_file = os.path.join(folder_path,file)
        decodeobj = decode(path_file)
        # print(decodeobj ,)
        try:
            bar_code = str(decodeobj[0].data)
            bar_code = bar_code[2:-1]
            print(bar_code, "path of file:",path_file,  "\n")
            bar_code_list.append(bar_code)
        except IndexError:
            print("Bar code not found")
        decodeobj_op.append(decodeobj)
        if bar_code == 0:
            os.remove(file)
    unnamed = list(set(barcode_samples)-set(bar_code_list))
    print(bar_code_list)
    print(len(bar_code_list))    
    print(unnamed)
    return "Process done"