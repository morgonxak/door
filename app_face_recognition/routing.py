from flask import render_template, request, make_response
from PIL import Image
from io import BytesIO
import cv2
import base64
import numpy as np

from app_face_recognition import app

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', URL=app.config['ip_server'] + ':' + str(app.config['port']))

################################################################################################
@app.route('/pull/', methods=['GET', 'POST'])
def receipt_image_from_devices():  ## todo Проверить что то тут не так
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
    if not app.config['debug']:
        print("push", json.dumps(personID))

    return str(personID)  # todo return json.dumps(personID)

