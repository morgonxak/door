import cv2
import dlib
import numpy as np
import os
#from skimage import io
import cv2
import time

from scipy.spatial import distance
import threading


class recognition_dlib:
    def __init__(self, path_data):
        path_predictor = os.path.join(path_data, r'shape_predictor_68_face_landmarks.dat')
        path_recognition_model = os.path.join(path_data, r'dlib_face_recognition_resnet_model_v1.dat')

        self.sp = dlib.shape_predictor(path_predictor)
        self.facerec = dlib.face_recognition_model_v1(path_recognition_model)

        self.detector = dlib.get_frontal_face_detector()

    def shape_to_np(self, shape, dtype="int"):
        '''
        конвертирует объект dlib spape в numpy array
        :param shape:
        :param dtype:
        :return:
        '''
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates

        return coords

    def rect_to_bb(self, rect):
        '''
        преобразует точки лица из формата dlib в cv2
        :param rect:
        :return:
        '''
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def get_detector_face(self, image):
        '''
        принимает изобрадения и возвращает массив с координатами найденных лиц
        :param image:
        :return:
        '''
        det = self.detector(image, 1)
        return det

    def get_spape(self, image, detectop):
        '''
        Определяем граници лица по детектору и изображению
        :param image:
        :param detectop:
        :return:
        '''
        shape = self.sp(image, detectop)
        return shape

    def get_descriptor(self, image, shape):
        '''
        Определяем дескриптор
        :param image:
        :param shape:
        :return:
        '''
        fase_description = self.facerec.compute_face_descriptor(image, shape)
        return fase_description

    def test(self, image):
        det = self.get_detector_face(image)
        if len(det) > 0:
            det_id = det[0]
            shape = self.get_spape(image, det_id)
            fase_description = self.get_descriptor(image, shape)

            shape_cv2 = self.shape_to_np(shape)
            faces = [self.rect_to_bb(det_id)]  #(x, y, w, h)

            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # for (x, y) in shape_cv2:
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #cv2.imshow("Output", image)
            #cv2.waitKey(0)
            return shape_cv2, fase_description, faces
        else:
            return None, None, None

def main_1():
    path_dir = r'C:\Users\cmitd\PycharmProjects\cameraTest\data'

    path_images = r'C:\Users\cmitd\PycharmProjects\faceId\face\roma\special_points'
    list_file_image = os.listdir(path_images)
    test = recognition_dlib(path_dir)

    for image in list_file_image:
        try:
            path_save_res = os.path.join(path_images, image[:image.rfind('.')] + '.txt')
            print(path_save_res)
            input = os.path.join(path_images, image)
            image_np = cv2.imread(input)
            file_Res, d = test.test(image_np)
        except BaseException:
            pass
        else:
            np.savetxt(path_save_res, file_Res, delimiter=' ', fmt='%s')


if __name__ == '__main__':
    main_1()
