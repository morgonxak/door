'''
Модуль для утравления реле для открытия и закрытия двери
также при нажатии кнопки открывается дверь
'''
try:
    import board
    import digitalio
except BaseException as e:
    print("error load RPI.GPIO {}".format(e))
    debug = True
else:
    debug = False

import threading
import time

class periphery(threading.Thread):
    def __init__(self, openTimeOut=3):
        '''

        :param openTimeOut: Время открытия двери после нажатия кнопки или получения сигнала на открытия двери
        '''
        super().__init__()

        self.openTimeOut = openTimeOut

        self.old_time = 0
        self.opened_door = False
        self.old_time = time.time()
        self.rele = digitalio.DigitalInOut(board.D18)
        self.rele.direction = digitalio.Direction.OUTPUT

        self.button = digitalio.DigitalInOut(board.D4)
        self.button.direction = digitalio.Direction.INPUT


    def __open_door(self):
        '''

        :return:
        '''

        self.rele.value = False
        #print("open door")

    def __close_door(self):
        '''

        :return:
        '''
        self.rele.value = True
        #print("close door")

    def is_door(self):
        '''
        Возвращает состояние двери
        :return:
        '''
        return self.opened_door

    def open_door(self):
        '''
        Открытия двери не по кнопке
        :return:
        '''
        self.opened_door = True
        self.old_time = time.time()
        #self.__open_door()

    def close_door(self):
        '''
        Закрытия двери не по кнопке
        :return:
        '''
        self.opened_door = False
        self.old_time = time.time()
        #self.__close_door()


    def run(self):
        '''
        :return:
        '''
        time_old = time.time()
        while True:

            key = self.button.value

            if key:
                if time.time() - time_old >= 0.5:

                    #print("Нажата кнопка на открытия двери", self.opened_door)
                    self.opened_door = True
                    self.old_time = time.time()
            else:
                time_old = time.time()

            if self.opened_door:
                if time.time() - self.old_time >= self.openTimeOut:
                    self.__close_door()
                    self.opened_door = False
                    print(self.opened_door)
                else:
                    self.__open_door()



if __name__ == '__main__':
    rele_pin = 9
    button_pin = 25
    test = periphery(rele_pin, button_pin)
    test.start()
    print("start")



