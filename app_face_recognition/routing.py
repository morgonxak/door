from flask import Flask, render_template, Response, request,make_response
from PIL import Image
from io import BytesIO
import cv2
import base64
import numpy as np

from app_face_recognition import app

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

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

    return render_template('index.html')

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

