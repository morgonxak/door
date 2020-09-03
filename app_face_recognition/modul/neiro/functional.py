'''
Модуль распознования функционал
'''
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import os
from keras import backend as K


def create_wrong_rgbd(file_path):
    '''
    берет 2 фотографии с разных пользователей
    :param file_path:
    :return:

    Получаем RGB + стредненное dep
    '''
    # folder = np.random.choice(glob.glob(file_path + "*"))
    folder_users = os.listdir(file_path)

    folder = os.path.join(file_path, np.random.choice(folder_users))
    print(folder)

    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0

    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    print(depth_file)

    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue

                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)

    mat_small = mat[140:340, 220:420]

    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(mat_small)
    # plt.show()

    # folder2 = np.random.choice(glob.glob(file_path + "*"))
    folder2 = os.path.join(file_path, np.random.choice(folder_users))

    while folder == folder2 or folder2 == "datalab":  # it activates if it chose the same folder
        folder2 = os.path.join(file_path, np.random.choice(folder_users))


    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0

    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:

                if val == "\n": continue

                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)

    mat2_small = mat2[140:340, 220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640, 480))
    img2 = np.asarray(img2)
    img2 = img2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)

    # plt.imshow(img2)
    # plt.show()
    # plt.imshow(mat2_small)
    # plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    full2 = np.zeros((200, 200, 4))
    full2[:, :, :3] = img2[:, :, :3]
    full2[:, :, 3] = mat2_small
    return np.array([full1, full2])

def create_couple_rgbd(file_path):
    '''
    Берет 2 фотографии у одного и того же пользователя
    :param file_path:
    :return:
    '''
    folder_users = os.listdir(file_path)

    folder = os.path.join(file_path, np.random.choice(folder_users))

    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    #    plt.imshow(mat_small)
    #    plt.show()
    #    plt.imshow(img)
    #    plt.show()

    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    img2 = Image.open(depth_file[:-5] + "c.bmp")
    img2.thumbnail((640, 480))
    img2 = np.asarray(img2)
    img2 = img2[160:360, 240:440]

    #   plt.imshow(img2)
    #   plt.show()
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #   plt.imshow(mat2_small)
    #   plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    full2 = np.zeros((200, 200, 4))
    full2[:, :, :3] = img2[:, :, :3]
    full2[:, :, 3] = mat2_small
    return np.array([full1, full2])

def create_couple(file_path):
    #folder = np.random.choice(glob.glob(file_path + "*"))

    folder_users = os.listdir(file_path)

    folder = os.path.join(file_path, np.random.choice(folder_users))

    while folder == "datalab":
        folder = np.random.choice(glob.glob(file_path + "*"))
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    #    plt.imshow(mat_small)
    #    plt.show()

    mat2 = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat2[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat2 = np.asarray(mat2)
    mat2_small = mat2[140:340, 220:420]
    mat2_small = (mat2_small - np.mean(mat2_small)) / np.max(mat2_small)
    #    plt.imshow(mat2_small)
    #    plt.show()
    return np.array([mat_small, mat2_small])

def create_input_rgbd(file_path):
    #  print(folder)
    mat = np.zeros((480, 640), dtype='float32')
    i = 0
    j = 0
    depth_file = file_path
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val = 1200
                mat[i][j] = float(int(val))
                j += 1
                j = j % 640

            i += 1
        mat = np.asarray(mat)
    mat_small = mat[140:340, 220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640, 480))
    img = np.asarray(img)
    img = img[140:340, 220:420]
    mat_small = (mat_small - np.mean(mat_small)) / np.max(mat_small)
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mat_small)
    #plt.show()
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    #plt.show()

    full1 = np.zeros((200, 200, 4))
    full1[:, :, :3] = img[:, :, :3]
    full1[:, :, 3] = mat_small

    return np.array([full1])

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


if __name__ == '__main__':
    res = create_couple(r'C:\Users\admin\PycharmProjects\testNeiro_faceID\faceid_train')
    print(res)
    print(len(res[0]))