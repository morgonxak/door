from flask import Flask

import cv2
import pickle
import os
from app_face_recognition.modul.processing_faceId import processing_faceid
from app_face_recognition.modul.Seek_termal import Thermal
from app_face_recognition.modul.periphery import periphery
import queue

app = Flask(__name__, static_url_path='/static')

app.config['queue'] = queue.Queue()
#
k = 1
pathProject_book = r'/home/dima/PycharmProjects'
pathProject_RPi = r'/home/pi/project'
pathProject_jetson = r'//home/dima/project'
#
if os.path.isdir(pathProject_book):
    pathProject = pathProject_book
else:
    pathProject = pathProject_jetson#pathProject_RPi

path_cvm_model = os.path.join(pathProject, 'faseid_door/rs/svm_model_{}.pk'.format(k))
path_knn_model = os.path.join(pathProject, 'faseid_door/rs/knn_model_{}.pk'.format(k))
path_classificator = os.path.join(pathProject, 'faseid_door/rs/haarcascade_frontalface_default.xml')
path_model_neiro = os.path.join(pathProject, 'faseid_door/rs/dlib/model_1.0.h5')
#
model_cvm = pickle.load(open(path_cvm_model, 'rb'))
model_knn = pickle.load(open(path_knn_model, 'rb'))
face_detector = cv2.CascadeClassifier(path_classificator)



# app.config['IR_camera'] = Thermal()
# try:
#     app.config['IR_camera'].initialize()
# except BaseException as e:
#     print("init IR camera: {}".format(e))
# else:
#     app.config['IR_camera'].start()
#     print("init IR camera - ok")

# app.config['door'] = periphery()
# app.config['door'].start()

app.config['IR_camera'] = None
app.config['door'] = None
app.config['dict_res'] = {}

app.config['recognition'] = processing_faceid(app.config['queue'], model_cvm, model_knn, face_detector, path_model_neiro, app.config['door'], app.config['dict_res'], app.config['IR_camera'])
app.config['recognition'].start()


from app_face_recognition import routing



