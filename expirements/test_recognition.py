from app_face_recognition.modul import cameraRealSense
import pickle
import cv2
# import face_recognition
from app_face_recognition.modul.neiro import Neiro_face
#
model_cvm = pickle.load(open(r'/home/dima/PycharmProjects/faseid_door/rs/svm_model_1.pk', 'rb'))
model_knn = pickle.load(open(r'/home/dima/PycharmProjects/faseid_door/rs/knn_model_1.pk', 'rb'))
face_detector = cv2.CascadeClassifier(r'/home/dima/PycharmProjects/faseid_door/rs/haarcascade_frontalface_default.xml')

def predict_cvm(face_encoding):
    '''
    Проверяет пользователя по модели CVM
    :param face_encoding: Получаем дескриптор
    :return: person_id -- Уникальный идентификатор пользователя
    '''
    # Прогнозирование всех граней на тестовом изображении с использованием обученного классификатора

    person_id = model_cvm.predict(face_encoding)

    return person_id

def predict_knn(face_encoding, tolerance=0.4):
    '''
    Проверяет пользователя по модели knn
    :param face_encoding:
    :param tolerance: Коэфициент похожести
    :return: person_id, dist == уникальный идентификатор и дистанция до него
    '''
    closest_distances = model_knn.kneighbors(face_encoding, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= tolerance for i in range(1)]

    if are_matches[0]:
        person_id = model_knn.predict(face_encoding)[0]
    else:
        person_id = "Unknown"

    return person_id


def getFace(frame_RGB, image_RGBD):
    gray = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    fases_200_200 = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_RGB, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("fase", frame_RGB)
        fase = frame_RGB[y:y+w,x:x+h]
        fase_RGBD = image_RGBD[y:y+w,x:x+h]

        fase = cv2.resize(fase, (200,200))
        fase_RGBD = cv2.resize(fase_RGBD, (200,200))

        cv2.imshow("fase_local", fase)
        fases_200_200.append(fase_RGBD)

    return fases_200_200



def test_1():
    flag = False
    rs = cameraRealSense()
    path_madel =r'/home/dima/PycharmProjects/faseid_door/rs/dlib/model_1.0.h5'
    neiro = Neiro_face(path_madel)
    descriptor_fase_1 = None
    for color_image, image_RGBD in rs.getFrame():
        cv2.imshow("color", color_image)

        face_locations = getFace(color_image, image_RGBD)

        if flag:

            for face_200_200_rgbd in face_locations:
                #print(face_200_200_rgbd.shape)
                descriptor_fase_2 = face_200_200_rgbd

                dist = neiro.get_predictions([descriptor_fase_1], [descriptor_fase_2])
                if dist < 0.4:
                    print("dima", dist)
                else:
                    print("no")


        key = cv2.waitKey(1)

        if key == ord("r"):
            flag = True
            for face_200_200_rgbd in face_locations:
                # print(face_200_200_rgbd.shape)
                descriptor_fase_1 = face_200_200_rgbd

        if key == ord("q"):
            break

if __name__ == '__main__':
    test_1()
