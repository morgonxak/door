from flask import Flask, render_template, Response, request,make_response, jsonify
from PIL import Image
from io import BytesIO
import cv2
import base64
import numpy as np

# from app_thermometer.moduls.camera import VideoCamera
from app_face_recognition import app

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

################################################################################################

def gen_video():
    """Video streaming generator function."""
    while True:
        frame = app.config['rs'].get_frame_rgb_by_web()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################################################

def gen_video_ir():
    while True:
        frame = app.config['recognition'].getImageIR()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_ir_feed')
def video_ir_feed():
    return Response(gen_video_ir(), mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################################################
def gen_video_frame_rgbd():
    while True:
        frame = app.config['rs'].get_frame_rgbd_by_web()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_frame_rgbd_feed')
def video_frame_rgbd_feed():
    return Response(gen_video_frame_rgbd(), mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################################################

def gen_video_bg():
    while True:
        frame = app.config['rs'].get_frame_rgbd_bg_removed_by_web()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_bg_feed')
def video_bg_feed():
    return Response(gen_video_bg(), mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################################################

def gen_found_people():
    while True:
        found_people = ['adasd','asd']#app.config['recognition'].get_found_people()
        yield #(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +  + b'\r\n')  #todo

@app.route('/found_people_feed', methods=['GET', 'POST'])
def found_people_feed():
    return str(app.config['recognition'].get_found_people())


################################################################################################
@app.route('/pull/', methods=['GET', 'POST'])
def receipt_image_from_devices():
    '''
    Принимает изображения с устройства
    :param id_devices:
    :return:
    '''
    if request.method == 'POST':
        req_data = request.get_data()
        try:
            im = Image.open(BytesIO(base64.b64decode(req_data[22:])))
        except BaseException as e:
            print("Error: " + str(e))
            return "Error: " + str(e)

        opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)

        app.config['queue'].put(opencvImage)


        return make_response("<h2>404 Error</h2>", 200)

    return render_template('index.html', URL=app.config['IP_Server'] + ':' + str(app.config['PORT_server']))

import json
@app.route('/get_res/', methods=['GET', 'POST'])
def pull_res_from_devices():
    '''
    Как появится результат отдает результат
    :param id_devices:
    :return:
    '''
    personID = app.config['recognition'].get_face_web()
    # if len(personID) != 0:
    #     personID['temp_face'] = 36.3
    print("jdbasjbdkas",personID)
    # return json.dumps(personID)
    # personID = {'eb8aa305-76a7-4c41-8d64-90fb765e7ad9': {'bbox': [150, 116, 223, 223],
    #                                           'name': 'Тимофеев Антон Евгеньевич::Главный менеджер проектов', 'temp': 36.5 },
    #  'b9e1d2bc-ac4f-4b5b-bec3-fd586c8c3e49': {'bbox': [436, 219, 185, 185],
    #                                           'name': 'Шумелев Дмитрий Игоревич::Программист 1 категории', 'temp': 36.7}}
    return str(personID)

@app.route('/get_temp/', methods=['GET', 'POST'])
def pullTemp():
    '''
    Как появится результат отдает результат
    :param id_devices:
    :return:
    '''
    personID = {'t': 36.6}

    # print("temp",personID)

    return str(json.dumps(personID))

