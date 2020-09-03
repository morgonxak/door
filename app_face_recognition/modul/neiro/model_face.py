'''
Теперь мы создаем сеть. Сначала мы создаем констративную потерю, затем определяем сетевую архитектуру,
начиная с архитектуры SqueezeNet, а затем используем ее в качестве сиамской сети для встраивания граней в коллектор.
'''

from keras.models import  Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Input, BatchNormalization, concatenate

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import models
from scipy.spatial import distance

#from tensorflow_core.keras import model

class Neiro_face():
    '''
    Класс предназначен для получения 128 признаков лица человека по фотографии
    '''
    def __init__(self, path_model: str):

        self.model_final = self.create_model()

        self.freeze(self.model_final)
        # Загрузка можеди
        self.model_final.load_weights(path_model)

    def freeze(self, model):
        """Freeze model weights in every layer."""
        for layer in model.layers:
            layer.trainable = False

            if isinstance(layer, models.Model):
                self.freeze(layer)

    def contrastive_loss(self, y_true, y_pred):
        '''
        Функция потерь
        :param y_true:
        :param y_pred:
        :return:
        '''
        margin = 1.
        return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

    def fire(self, x, squeeze=16, expand=64):

        x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
        x = Activation('relu')(x)

        left = Convolution2D(expand, (1, 1), padding='valid')(x)
        left = Activation('relu')(left)

        right = Convolution2D(expand, (3, 3), padding='same')(x)
        right = Activation('relu')(right)

        x = concatenate([left, right], axis=3)
        return x

    ###################################################################### Прорая модель  ##########################################################
    def create_model(self):
        img_input = Input(shape=(200, 200, 4))

        x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = self.fire(x, squeeze=16, expand=16)
        x = self.fire(x, squeeze=16, expand=16)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = self.fire(x, squeeze=32, expand=32)
        x = self.fire(x, squeeze=32, expand=32)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = self.fire(x, squeeze=48, expand=48)
        x = self.fire(x, squeeze=48, expand=48)
        x = self.fire(x, squeeze=64, expand=64)
        x = self.fire(x, squeeze=64, expand=64)
        x = Dropout(0.2)(x)
        x = Convolution2D(512, (1, 1), padding='same')(x)
        out = Activation('relu')(x)

        modelsqueeze = Model(img_input, out)
        modelsqueeze.summary()

        im_in = Input(shape=(200, 200, 4))
        x1 = modelsqueeze(im_in)
        x1 = Flatten()(x1)
        x1 = Dense(512, activation="relu")(x1)
        x1 = Dropout(0.2)(x1)
        feat_x = Dense(128, activation="linear")(x1)
        feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

        model_top = Model(inputs=[im_in], outputs=feat_x)
        model_top.summary()

        im_in1 = Input(shape=(200, 200, 4))
        im_in2 = Input(shape=(200, 200, 4))
        feat_x1 = model_top(im_in1)
        feat_x2 = model_top(im_in2)

        lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

        # model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)
        model_final = Model(inputs=[im_in1], outputs=feat_x1)
        model_final.summary()

        adam = Adam(lr=0.001)
        sgd = SGD(lr=0.001, momentum=0.9)

        model_final.compile(optimizer=adam, loss=self.contrastive_loss)
        return model_final

    def get_signs(self, img_RGBD):
        '''
        Получает на вход ФОТОГРАФИЮ Отдает 128 вризнаков
        :param img_RGBD:
        :return:
        '''
        signs = self.model_final.predict([img_RGBD])
        return signs

    def get_predictions(self, img_1_RGBD, img_2_RGBD):
        '''
        Функция определения предсказания один и тот же человек на фотографии или нет
        :param img_1_RGBD:
        :param img_2_RGBD:
        :return:
        '''
        signs_1 = self.model_final.predict([img_1_RGBD])
        signs_2 = self.model_final.predict([img_2_RGBD])

        return distance.euclidean(signs_1, signs_2)

if __name__ == '__main__':

    path_madel = r'C:\Users\admin\PycharmProjects\photoBP\rc\model_1.0.h5'
    file1 = (r'C:\Users\admin\PycharmProjects\testNeiro_faceID\faceid_val\(2012-05-18)(154728)\004_3_d.dat')
    file2 = (r'C:\Users\admin\PycharmProjects\testNeiro_faceID\faceid_val\(2012-05-18)(155357)\001_1_d.dat')

    simple = Neiro_face(path_madel)

    img_1_RGBD = create_input_rgbd(file1)
    img_2_RGBD = create_input_rgbd(file2)

    signs = simple.get_signs(img_1_RGBD)
    print("*"*70)
    print(signs)
    print("*"*70)
    predict = simple.get_predictions(img_1_RGBD, img_2_RGBD)
    print("Предсказания: {}".format(predict))
