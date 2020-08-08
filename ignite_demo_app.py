import json
from ignite_demo import *
from flask import Flask, request, render_template, jsonify
from azure.storage.blob import BlockBlobService
import time
import os
import pandas as pd
from search_utils import *


try:
    from config import *

    print('getting variables from config.py')

except BaseException:
    BLOB_ACCOUNT_NAME = os.environ['BLOB_ACCOUNT_NAME']
    BLOB_ACCOUNT_PASS = os.environ['BLOB_ACCOUNT_PASS']
    BLOB_CONTAINER_NAME = os.environ['BLOB_CONTAINER_NAME']
    print('getting environ variable')

# Blob service to get the files from
block_blob_service = BlockBlobService(
    account_name=BLOB_ACCOUNT_NAME,
    account_key=BLOB_ACCOUNT_PASS)

# blob container name
container_name = BLOB_CONTAINER_NAME


app = Flask(__name__, template_folder='templates')
App_path = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(App_path, 'temp')
if not os.path.exists(TEMP_PATH):
    os.mkdir(TEMP_PATH)
data_path = os.path.join(App_path, 'data')


def make_df(filepath='data/ignite_demo/catalog.xlsx'):
    df = pd.read_excel(filepath)
    df['long description'] = df[df.columns[1:-1]].apply(lambda x: '$$$'.join(x.dropna().astype(str)), axis=1)
    df = df[['Image', 'long description']]

    return df


@app.route('/', methods=["POST", "GET"])
def home():
    return render_template('home.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    try:
        clean_one_dir('static')
        makedirs()
    except BaseException:
        pass
    global graph

    static_path = pjoin(App_path, 'static')
    input_dir = pjoin(static_path, 'input')
    try:
        upload = request.files.getlist("file")[0]
    except IndexError:
        return render_template('home.html')
    print(upload.filename)
    filename = upload.filename
    if not allowedfile(filename):
        return render_template('home.html')
    filepath = pjoin(input_dir, filename)
    upload.save(filepath)
    localbool = 1
    with graph.as_default():
        results = driver_crop_image(filepath, 'static/crops', filename, localbool)
    for each_result in results:
        each_result['out_img'] = ['output/' + e for e in each_result['out_img']]
    return render_template('ignite_dyn.html', infile='input/' + filename, seq=results)


@app.route('/decoder', methods=["POST", "GET"])
def decoder():
    if len(os.listdir('static')) != 0:
        cleandir()
        makedirs()

    try:
        upload = request.files.getlist("file")[0]
    except IndexError:
        return render_template('home.html')
    print(upload.filename)
    static_path = pjoin(App_path, 'static')
    input_dir = pjoin(static_path, 'input')
    filename = upload.filename
    if not allowedfile(filename):
        return render_template('home.html')
    filepath = pjoin(input_dir, filename)
    upload.save(filepath)
    decodeobj = decode(filepath)
    if len(decodeobj) == 0:
        decodeobj = [{'type': 'Barcode not found in image', 'data': 'Barcode not found in image'}]
    return render_template('decoder.html', infile='input/' + filename, obj=decodeobj)


@app.route('/blob_upload', methods=["POST", "GET"])
def blob_upload():
    global graph
    folder_name_inst = pjoin('temp', 'temp_' + str(time.time()).replace(".", "_"))
    in_path, out_path, _, detect_path, crops_path = makedirs_uniq(folder_name_inst)
    got_json = request.json
    '''
    Sample input Json
    {
    "ZipId":"1555658192865",
    "FileName":"gloves100.jpg",
    "BpcCode":99999,
    "UserId":0
    }
    '''
    try:
        zip_id = got_json['ZipId']
        BPC_codes = got_json['BpcCode']
        filename = got_json['FileName']
        # Rkey_list = got_json['FileList']
        user_id = got_json['UserId']
        # user_name = got_json['UserName']
        for key, val in got_json.items():
            if not val:
                return jsonify({"error": "Missing " + str(key)}), 400
    except KeyError:
        clean_one_dir(folder_name_inst)
        return jsonify({"error": "Missing required parameters"}), 400
    except TypeError:
        return jsonify({"error": "Missing required parameters"}), 400
    blob_file_path = "bpc/" + str(BPC_codes) + '/' + str(zip_id) + '/image/' + filename
    print(blob_file_path)
    local_filepath = pjoin(in_path, filename)
    try:
        block_blob_service.get_blob_to_path(
            container_name, blob_file_path, local_filepath)
    except BaseException:
        clean_one_dir(folder_name_inst)
        return jsonify({"error": "File not found in Blob"}), 400
    with graph.as_default():
        results = driver_crop_image(local_filepath, crops_path, filename, localbool=0, out_path=out_path)
    for each in results:
        crop_filename = each['crop_img']
        blob_crop_path = "bpc/" + str(BPC_codes) + '/' + str(zip_id) + '/' + crop_filename
        # uploading file to blob
        block_blob_service.create_blob_from_path(container_name, blob_crop_path, pjoin(folder_name_inst, crop_filename))
    clean_one_dir(folder_name_inst)
    return jsonify(results), 200


@app.route('/blob_decoder', methods=["POST", "GET"])
def blob_decoder():
    got_json = request.json
    if len(os.listdir('static')) != 0:
        cleandir()
    try:
        zip_id = got_json['ZipId']
        BPC_codes = got_json['BpcCode']
        filename = got_json['FileName']
        # Rkey_list = got_json['FileList']
        user_id = got_json['UserId']
        # user_name = got_json['UserName']
        for key, val in got_json.items():
            if not val:
                return jsonify({"error": "Missing " + str(key)}), 400
    except KeyError:
        return jsonify({"error": "Missing required parameters"}), 400
    except TypeError:
        return jsonify({"error": "Missing required parameters"}), 400
    folder_name_inst = pjoin('temp', 'temp_' + str(time.time()).replace(".", "_"))
    in_path, out_path, _, detect_path, crops_path = makedirs_uniq(folder_name_inst)

    '''
    Sample input Json
    {
    "ZipId":"1555658192865",
    "FileName":"gloves100.jpg",
    "BpcCode":99999,
    "UserId":0
    }
    '''
    blob_file_path = "bpc/" + str(BPC_codes) + '/' + str(zip_id) + '/barcode/' + filename
    local_filepath = pjoin(in_path, filename)
    try:
        block_blob_service.get_blob_to_path(
            container_name, blob_file_path, local_filepath)
    except BaseException:
        return jsonify({"error": "File not found in Blob"}), 400
    with graph.as_default():
        results = decode(local_filepath)
    try:
        bar_code = str(results[0].data)
    except IndexError:
        return jsonify({"error": "Bar code not found"}), 400
    bar_code = bar_code[2:-1]
    clean_one_dir(folder_name_inst)
    df = make_df()
    res = search_text(bar_code, df)
    return jsonify({'input': bar_code, 'output': res}), 200


@app.route('/text_search', methods=["POST", "GET"])
def se_text():
    # print(make_df())
    got_json = request.json
    # got_json = {"text" : ""}
    inp_text = got_json["Text"]
    df = make_df()
    res = search_text(inp_text, df)
    return jsonify({'input': inp_text, 'output': res}), 200


def allowedfile(name):
    # print(name[-3:])
    return name.split()[-1].lower() in ['jpg', 'jpeg', 'png']


def makedirs_uniq(path):
    folders = ['input', 'output', 'inputfiles', 'detect', 'crops']
    if not os.path.exists(path):
        os.mkdir(path)
    for foldername in folders:
        if not os.path.exists(pjoin(path, foldername)):
            os.mkdir(pjoin(path, foldername))
    return pjoin(
        path, 'input'), pjoin(
        path, 'output'), pjoin(
            path, 'inputfiles'), pjoin(
                path, 'detect'), pjoin(
                    path, 'crops')


def makedirs():
    folders = ['input', 'output', 'inputfiles', 'detect', 'crops']
    if not os.path.exists('static'):
        os.mkdir('static')
    for foldername in folders:
        if not os.path.exists(pjoin('static', foldername)):
            os.mkdir(pjoin('static', foldername))


def cleandir(foldername=''):
    try:
        shutil.rmtree('static/input')
        shutil.rmtree('static/output')
        shutil.rmtree('static/inputfiles')
        shutil.rmtree('static/detect')
        shutil.rmtree('static/crops')
    except BaseException:
        pass


def clean_one_dir(foldername):
    try:
        shutil.rmtree(foldername)
    except BaseException:
        pass


def api():
    app.run(debug=True, host="0.0.0.0", port=4050, threaded=True)


if __name__ == '__main__':
    api()
    # pass
