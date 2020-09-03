'''
Модуль для проверки распознования миц и для дообучения CVM, Knn нейронных сетей
'''
import threading
import face_recognition
import cv2
import numpy
import time
import os
from app_face_recognition.modul.periphery import periphery
from app_face_recognition.modul.dict_user import known_face_names


class processing_faceid(threading.Thread):

    def __init__(self, queue, model_cvm, model_knn, face_detector, path_madel, door:periphery, dict_res, IR_camera):
        super().__init__()
        self.queue, self.model_cvm, self.model_knn, self.face_detector, self.path_madel, self.door = queue, model_cvm, model_knn, face_detector, path_madel, door
        self.dict_res = {}
        self.T_img = None
        self.IR_camera = IR_camera

    def get_face_web(self):
        return self.dict_res

    def getFace(self, frame_RGB):
        '''
        Находим лицо и обрезаем
        :param frame_RGB:
        :param image_RGBD:
        :return:
        '''
        gray = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        fases_200_200 = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_RGB, (x, y), (x + w, y + h), (255, 0, 0), 2)
            fase_RGB_200_200 = frame_RGB[y:y + w, x:x + h]

            #fase_RGB_200_200 = cv2.resize(fase_RGB_200_200, (200, 200))

            # fase_RGBD_200_200 = None
            #cv2.imshow("fase_local", fase_RGB_200_200)
            fases_200_200.append([fase_RGB_200_200, (x, y, w, h)])

        return fases_200_200

    def get_descriptor_RGB(self, fase_RGB_200_200):
        '''
        создаем дискриптор для RGBизображения
        :param fase_RGB_200_200:
        :return:
        '''
        face_encoding = face_recognition.face_encodings(fase_RGB_200_200)
        return face_encoding

    def __predict_cvm(self, face_encoding):
        '''
        Проверяет пользователя по модели CVM
        :param face_encoding: Получаем дескриптор
        :return: person_id -- Уникальный идентификатор пользователя
        '''
        # Прогнозирование всех граней на тестовом изображении с использованием обученного классификатора
        try:
            person_id = self.model_cvm.predict(face_encoding)
        except ValueError:
            person_id = None

        return person_id

    def __predict_knn(self, face_encoding, tolerance=0.4):
        '''
        Проверяет пользователя по модели knn
        :param face_encoding:
        :param tolerance: Коэфициент похожести
        :return: person_id, dist == уникальный идентификатор и дистанция до него
        '''
        try:
            closest_distances = self.model_knn.kneighbors(face_encoding, n_neighbors=1)

            are_matches = [closest_distances[0][i][0] <= tolerance for i in range(1)]

            if are_matches[0]:
                person_id = self.model_knn.predict(face_encoding)[0]
            else:
                person_id = "Unknown"
        except ValueError:
            person_id = None

        return person_id


    def open_door(self):
        self.door.open_door()

    def get_temp(self):
        T_text_path = r'/home/dima/project/faseid_door/app_face_recognition/static/tMAXT.txt'
        T_image_path = r'/home/dima/project/faseid_door/app_face_recognition/static/Temp_thermal_image.png'
        count = 0
        temp, T_img = None, None
        while count <= 3:
            temp = -100
            try:
                if os.path.exists(T_image_path):
                    if os.path.exists(T_text_path):
                        f = open(T_text_path, 'r')
                        temp = float(f.read())
                        f.close()

                        if temp != -100:
                            if temp <= 32.5:
                                temp = temp + 4.3 + abs(32.5 - temp) / 1.23076923
                            else:
                                temp = temp + 4.3
                            print("на тепловизоре макс.: {}".format(temp))
                        else:
                            continue

                        T_img = cv2.imread(T_image_path, cv2.IMREAD_GRAYSCALE)
                        T_img = cv2.rotate(T_img, cv2.ROTATE_90_CLOCKWISE)  # rotateImage(Tt_img, 270)
                        T_img = T_img.astype(numpy.uint8)
                        T_img = cv2.cvtColor(T_img, cv2.COLOR_GRAY2RGB)
                        T_img = cv2.applyColorMap(T_img, cv2.COLORMAP_JET)


                        return temp, T_img

                else:
                    count += 1
                    continue
            except BaseException as e:
                count += 1
                continue

    def getImageIR(self):
        if not self.T_img is None:
            ret, jpeg = cv2.imencode('.jpg', self.T_img)
            return jpeg.tobytes()

        else:
            return None

    def run(self):
        '''
        Фходные данные Изображения в формате RGB
        :return:
        '''
        oldTime = time.time()
        while True:

            frame = self.queue.get()

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            face_locations = self.getFace(frame)

            # if time.time() - oldTime >= 2:
            #     temp, self.T_img = 1, 1#self.get_temp()
            #     oldTime = time.time()

            temp = 36.6

            dict_res = {}
            for count, (fase_RGB_200_200, (x, y, w, h)) in enumerate(face_locations):


                descriptor_fase_RGB = self.get_descriptor_RGB(fase_RGB_200_200)

                res_predict_cvm = self.__predict_cvm(descriptor_fase_RGB)
                res_predict_knn = self.__predict_knn(descriptor_fase_RGB)

                if res_predict_cvm == res_predict_knn:
                    if res_predict_cvm is None:
                        pass
                    else:
                        people = known_face_names.get(res_predict_cvm[0], None)
                        print("Найден пользователь:", people)
                        if not people is None:

                            # self.open_door()


                            if not res_predict_cvm[0] in dict_res:
                                dict_res[res_predict_cvm[0]] = {}
                                dict_res[res_predict_cvm[0]]['bbox'] = [x, y, w, h]
                                dict_res[res_predict_cvm[0]]['name'] = people
                                dict_res[res_predict_cvm[0]]['temp'] = round(temp, 1)
                            else:
                                dict_res[res_predict_cvm[0]]['bbox'] = [x, y, w, h]
                                dict_res[res_predict_cvm[0]]['name'] = people
                                dict_res[res_predict_cvm[0]]['temp'] = round(temp, 1)


                dict_res[str(count+1)] = {}
                dict_res[str(count+1)]['bbox'] = [x, y, w, h]
                dict_res[str(count+1)]['name'] = 'none'
                dict_res[str(count+1)]['temp'] = round(temp, 1)


            self.dict_res = dict_res

            # cv2.imshow("rgb", frame)
            # cv2.waitKey(1)






