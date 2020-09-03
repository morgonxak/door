from app_face_recognition.modul.camera_qwe import cameraRealSense

import cv2
import time
import pickle
import os
import numpy

# import face_recognition
from app_face_recognition.modul.neiro.model_face import Neiro_face
from app_face_recognition.modul.processing_faceId import processing_faceid
import face_recognition


#
k = 2
model_cvm = pickle.load(open(r'/home/dima/PycharmProjects/faseid_door/rs/svm_model_{}.pk'.format(k), 'rb'))
model_knn = pickle.load(open(r'/home/dima/PycharmProjects/faseid_door/rs/knn_model_{}.pk'.format(k), 'rb'))
face_detector = cv2.CascadeClassifier(r'/home/dima/PycharmProjects/faseid_door/rs/haarcascade_frontalface_default.xml')
path_madel =r'/home/dima/PycharmProjects/faseid_door/rs/dlib/model_1.0.h5'

rs = cameraRealSense()
neiro = Neiro_face(path_madel)
rec_rgb = processing_faceid()

def getFace(frame_RGB, image_RGBD):
    '''
    Находим лицо и обрезаем
    :param frame_RGB:
    :param image_RGBD:
    :return:
    '''
    gray = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    fases_200_200 = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_RGB, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("fase", frame_RGB)
        fase_RGB_200_200 = frame_RGB[y:y+w,x:x+h]
        fase_RGBD_200_200 = image_RGBD[y:y+w,x:x+h]

        fase_RGB_200_200 = cv2.resize(fase_RGB_200_200, (200,200))
        fase_RGBD_200_200 = cv2.resize(fase_RGBD_200_200, (200,200))

        cv2.imshow("fase_local", fase_RGB_200_200)
        fases_200_200.append([fase_RGB_200_200, fase_RGBD_200_200])

    return fases_200_200

def __predict_cvm(face_encoding):
    '''
    Проверяет пользователя по модели CVM
    :param face_encoding: Получаем дескриптор
    :return: person_id -- Уникальный идентификатор пользователя
    '''
    # Прогнозирование всех граней на тестовом изображении с использованием обученного классификатора
    try:
        person_id = model_cvm.predict(face_encoding)
    except ValueError:
        person_id = None

    return person_id

def __predict_knn(face_encoding, tolerance=0.4):
    '''
    Проверяет пользователя по модели knn
    :param face_encoding:
    :param tolerance: Коэфициент похожести
    :return: person_id, dist == уникальный идентификатор и дистанция до него
    '''
    try:
        closest_distances = model_knn.kneighbors(face_encoding, n_neighbors=1)

        are_matches = [closest_distances[0][i][0] <= tolerance for i in range(1)]

        if are_matches[0]:
            person_id = model_knn.predict(face_encoding)[0]
        else:
            person_id = "Unknown"
    except ValueError:
        person_id = None

    return person_id

def get_descriptor_RGB(fase_RGB_200_200):
    '''
    создаем дискриптор для RGBизображения
    :param fase_RGB_200_200:
    :return:
    '''
    face_encoding = face_recognition.face_encodings(fase_RGB_200_200)
    return face_encoding

def f(lst):
    elems = {}
    e, em = None, 0
    for i in lst:
        elems[i] = t = elems.get(i, 0) + 1
        if t > em:
            e, em = i, t
    return e

def test_1():
    predict_neiro = "NONE"
    rs = cameraRealSense()
    frames = []


    descriptor_fase_1 = None
    for color_image, image_RGBD in rs.getFrame():
        cv2.imshow("color", color_image)

        face_locations = getFace(color_image, image_RGBD)

        for fase_RGB_200_200, fase_RGBD_200_200 in face_locations:
            descriptor_fase_RGB = get_descriptor_RGB(fase_RGB_200_200)
            if not descriptor_fase_1 is None:
                oldTime = time.time()
                dist = neiro.get_predictions([descriptor_fase_1], [fase_RGBD_200_200])
                print("time neiro {}".format(time.time() - oldTime))

                if dist < 0.4:
                    predict_neiro = "dima"
                else:
                    predict_neiro = "no"

            oldTime = time.time()
            predict_cvm = __predict_cvm(descriptor_fase_RGB)
            print("time predict_cvm {}".format(time.time() - oldTime))

            if str(predict_cvm) == "['a2f925fb-9fd0-4294-a1c5-fc830334c649']":
                p_cvm = "dima"
            else:
                p_cvm = "no"

            if predict_neiro == p_cvm: print("*"*10)

            #frames.append(p_cvm)
            frames.append(predict_neiro)
            if len(frames) == 10:
                id_user = f(frames)
                frames.clear()
                print(id_user)



        key = cv2.waitKey(1)
        if key == ord("r"):
            for fase_RGB_200_200, fase_RGBD_200_200 in face_locations:

                descriptor_fase_1 = fase_RGBD_200_200

        if key == ord("q"):
            break

def loadImage_pickl(path):
    with open(path, 'rb') as f:
        list_photo = pickle.load(f)
    return list_photo

dict_people = {}
def load_image_people(pathPhoto):
    list_people = os.listdir(pathPhoto)
    print("Количество пользователей в базе:", len(list_people))
    for people in list_people:
        listPhoto = loadImage_pickl(os.path.join(pathPhoto, people, 'RGB/photo.pickl'))
        print("Количество фото", len(listPhoto))
        for color_image, image_RGBD in listPhoto:
            if not people in dict_people:
                dict_people[people] = []
            dict_people[people].append([color_image, image_RGBD])

            #cv2.imshow("RGB phohot 0", color_image)
            #cv2.imshow("RGBD phohot 0", image_RGBD)
            #cv2.waitKey()
    print("Загрузка данных завершена")

def check(image_fase_1):
    '''
    пробегает по всем пользователям и находит лучший результат совподения
    :param descriptor_fase_1:
    :return:
    '''
    list_res = []
    list_dist = []
    for people in dict_people:
        for color_image, image_RGBD in dict_people[people]:
            dist = neiro.get_predictions([image_fase_1], [image_RGBD])
            if dist < 0.4:
                list_res.append(people)
                list_dist.append(dist)

    try:
        min_dist_people = numpy.where(list_dist == numpy.amin(list_dist))
        print(min_dist_people)
    except ValueError as e:
        print("eroor {}".format(e))
        return 0

    return list_res[min_dist_people[0][0]]


def run():

    for color_image, image_RGBD in rs.getFrame():
        cv2.imshow("color", color_image)

        face_locations = getFace(color_image, image_RGBD)

        for fase_RGB_200_200, fase_RGBD_200_200 in face_locations:
            descriptor_fase_RGB = get_descriptor_RGB(fase_RGB_200_200)
            #descriptor_fase_RGBD = neiro.get_signs([fase_RGBD_200_200])

            #cv2.imshow("landmarks", face_landmarks_list)
            res_predict_cvm = __predict_cvm(descriptor_fase_RGB)
            res_predict_knn = __predict_knn(descriptor_fase_RGB)

            print()
            oldTime = time.time()
            res = 0#check(fase_RGBD_200_200)
            print("time check {}, n_rgbd {}, cvm {} knn {}".format(time.time() - oldTime, res, res_predict_cvm, res_predict_knn))

        k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
        if k == 27:
            break


def save_image():
    dict_res = {}  #{'personId:[[rgb],[rgbd]]}
    for people in dict_people:
        for color_image, image_RGBD in dict_people[people]:
            cv2.imshow("fase", color_image)

            k = cv2.waitKey() & 0xff  # 'ESC' для Выхода
            if k == ord("w"):
                dict_res[people] = []
                dict_res[people].append([color_image, image_RGBD])
            if k == 27:
                break

    with open('/home/dima/PycharmProjects/faseid_door/rs/filename.pickle', 'wb') as handle:
        pickle.dump(dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

def add_user(fase_RGB_200_200, fase_RGBD_200_200):
    personId = input("Введите имя пользователя: ")
    dict_people[personId] = []
    dict_people[personId].append([fase_RGB_200_200,fase_RGBD_200_200])


def load_pickle_dict(path):
    global dict_people
    dict_people = loadImage_pickl(path)

if __name__ == '__main__':
    pathPhoto = r'/home/dima/Документы/photoBR'
    path_pickl_file = '/home/dima/PycharmProjects/faseid_door/rs/filename.pickle'
    #load_pickle_dict(path_pickl_file)
    load_image_people(pathPhoto)
    #save_image()
    run()
