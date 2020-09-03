from flask import Flask

import cv2
import pickle
import os
from app_face_recognition.modul.processing_faceId import processing_faceid
from app_face_recognition.modul.periphery import periphery
import queue

app = Flask(__name__, static_url_path='/static')

app.config['queue'] = queue.Queue()
#
k = 1
pathProject_book = r'/home/dima/PycharmProjects'
pathProject_jetson = r'//home/dima/project'
app.config['debug'] = True

#
if os.path.isdir(pathProject_book):
    pathProject = pathProject_book
else:
    pathProject = pathProject_jetson

path_cvm_model = os.path.join(pathProject, 'faseid_door/rs/svm_model_{}.pk'.format(k))
path_knn_model = os.path.join(pathProject, 'faseid_door/rs/knn_model_{}.pk'.format(k))
path_classificator = os.path.join(pathProject, 'faseid_door/rs/haarcascade_frontalface_default.xml')
path_model_neiro = os.path.join(pathProject, 'faseid_door/rs/dlib/model_1.0.h5')
#
model_cvm = pickle.load(open(path_cvm_model, 'rb'))
model_knn = pickle.load(open(path_knn_model, 'rb'))
face_detector = cv2.CascadeClassifier(path_classificator)

if not app.config['debug']:
    app.config['door'] = periphery()
    app.config['door'].start()
else:
    app.config['door'] = None


app.config['dict_res'] = {}

app.config['recognition'] = processing_faceid(app.config['queue'], model_cvm, model_knn, face_detector, path_model_neiro, app.config['door'])
app.config['recognition'].start()


from app_face_recognition import routing



