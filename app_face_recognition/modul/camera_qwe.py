'''
переделать
Переосмыслить переменную и убрать
self.status_init = False
Исправить ошибку при том если файл не записан а его хотят воспроизвести то вылазеет ошиобка при инициализации файла
 a2f925fb-9fd0-4294-a1c5-fc830334c649

'''

import pyrealsense2 as rs
import numpy as np
import cv2
import gc
import threading

class camera:
    '''
    Клас для получения данных с камеры RGB
    '''
    def __init__(self, URL):
        self.cam = cv2.VideoCapture(URL)

    def getFrame(self):
        while True:
            ret, img = self.cam.read()
            if ret:
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield ret, img
            k = cv2.waitKey(10) & 0xff  # 'ESC' для Выхода
            if k == 27:
                break

class cameraRealSense(threading.Thread):

    def __init__(self, pathCalsssificator):
        super().__init__()
        self.pathCalsssificator = pathCalsssificator
        self.status_recording = False  #Состояния записи
        self.status_stream = False  #отображения

        self.clipping_distance_in_meters = 1  # 1 meter
        #Объекты для выравнивания фотографий
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        #Создаем камеру:
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.profile = self.pipeline.start(self.config)
        self.device_obj = self.profile.get_device()

        depth_sensor = self.device_obj.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # Получаем разрешения камеры
        self.clipping_distance = self.clipping_distance_in_meters / depth_scale

        self.face_detector = cv2.CascadeClassifier(self.pathCalsssificator)

        self.status_stream = True

        self.color_image, self.full = [], []

    def stop_camera(self):
        '''
        Останавливает камеру либо для типа чтения с файла или с RS
        :param type_init:
        :return:
        '''

        self.status_stream = False
        self.pipeline.stop()

        # # #Удаляем объекты созданные при инициализации
        del self.config
        del self.pipeline

    def get_frame_rgb_by_web(self):
        '''
        отдает изобразения для гинератора на веб страницу
        :return:
        '''

        gray = cv2.cvtColor(self.color_image_by_web, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.color_image_by_web, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', self.color_image_by_web)

        return jpeg.tobytes()

    def get_frame_rgbd_by_web(self):
        '''
        отдает изобразения для гинератора на веб страницу
        :return:
        '''
        ret, jpeg = cv2.imencode('.jpg', self.depth_image)

        return jpeg.tobytes()

    def get_frame_rgbd_bg_removed_by_web(self):
        '''
        отдает изобразения для гинератора на веб страницу
        :return:
        '''
        ret, jpeg = cv2.imencode('.jpg', self.depth_image_bg_removes)
        return jpeg.tobytes()

    def getFrame_realsense(self):
        while self.status_stream:
            yield self.color_image, self.full

    def run(self):
        '''
        Генератор кадров глубины и RGB
        :return:
        '''
        while self.status_stream:
            frames = self.pipeline.wait_for_frames()
            # aligned_frames = self.align.process(frames)

            # Получить выровненные кадры
            #depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            # color_frame = aligned_frames.get_color_frame()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # full = np.zeros((480, 640, 4))
            # full[:, :, :3] = color_image[:, :, :3]
            #
            # mat_small = (depth_image - np.mean(depth_image)) / np.max(depth_image)
            # full[:, :, 3] = mat_small

            #Обрезаем фото по глубине
            grey_color = 0
            # depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels

            # bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            #Rotate
            # depth_colormap = np.rot90(depth_colormap, 3)
            # bg_removed = np.rot90(bg_removed, 3)
            color_image = np.rot90(color_image, 3)
            # full = np.rot90(full, 3)

            #Делаем копию для веба
            # self.depth_image = np.copy(depth_colormap)
            # self.depth_image_bg_removes = np.copy(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB))
            self.color_image_by_web = np.flipud(np.copy(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)))

            # self.color_image, self.full = np.copy(color_image), np.copy(full)
            self.color_image = np.copy(color_image)
            # yield color_image, full

    def __del__(self):
        gc.collect()


