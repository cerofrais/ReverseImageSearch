
from __init__ import *

from test import *
from yolov3_first import *

app = Flask(__name__,template_folder='templates')
App_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(App_path,'data')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=["POST","GET"])
def upload():
    if len(os.listdir('static')) != 0:
        cleandir()
    global graph
    with graph.as_default():
        if not os.path.exists('static/input'):
            os.mkdir('static/input')
        if not os.path.exists('static/output'):
            os.mkdir('static/output')
        if not os.path.exists('static/inputfiles'):
            os.mkdir('static/inputfiles')
        static_path = pjoin(App_path, 'static')
        input_dir = pjoin(static_path,'input')
        #for upload in request.files.getlist("file"):
        #print(request)
        upload = request.files.getlist("file")[0]

        print(upload.filename)
        filename = upload.filename
        if not allowedfile(filename):
            return render_template('home.html')
        filepath = pjoin(input_dir,filename)
        # print(zz)
        upload.save(filepath)
        yolov3_detect(filepath)
        results = []

        for each in os.listdir('static/inputfiles'):
            result = {}
            result ['in_img'] = 'inputfiles/'+each
            result ['out_img'] = ['output/'+e for e in get_names(get_neighbors(knn,pjoin('static/inputfiles',each)))]
            results.append(result)
            # result[each] = get_names(get_neighbors(knn,pjoin('inputfiles',each)))
        print(results)
        # result = get_names(get_neighbors(knn, filepath))
        # result1 = result['crop0.jpg']
        # return render_template('dynamic.html', img_1 = 'input/'+filename, img_2 = 'output/'+result1[0],\
        #                        img_3 = 'output/'+result1[1],img_4 = 'output/'+result1[2], strout = " \n ".join(result))
        return render_template('dyn.html', infile= 'input/'+filename, seq = results)

def allowedfile(name):
    #print(name[-3:])
    return name[-3:] in ['jpg','jpeg','png']

def cleandir():
    try :
        shutil.rmtree('static/input')
        shutil.rmtree('static/output')
        shutil.rmtree('static/inputfiles')
    except:
        pass


def api():
  app.run(debug = True, host = "0.0.0.0", port = 4040)

if __name__ == '__main__':
  api()
