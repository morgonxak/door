# Дверь [3.09.2020]
Проект для идентификации пользователей и отктытия двери для Института цифры.
## Основные пакеты:
0. threading
0. face_recognition
0. cv2
0. numpy
0. sklearn
0. flask
0. base64

## Установка:
0. Создать виртуальное окружения python3 -m venv door, активировать source door/bin/activate
0. pip install requirements.txt
0. поменять путь до папки с проектом в файле app_face_recognition/__init__.py в переменной pathProject_book поменять путь на свой.
0. Запуск: pyhton run_web.py

## Debur режим 
Включается в app_face_recognition/__init__.py переменная app.config['debug'] = True.
Если True - Включен, вместо открывания двери пишет сообщения в логи
Если False - Открывает дверь (Работает только в релизе)

## Описание папок проекта
expirements - Тестовые файлы.

rs - ресурсы проекта (содержат обученные модели и классификатор для поискаа лиц).

app_face_recognition - Основное приложения 
1. modul - Основные компоненты проекты
2. static - статика для Web страници
3. templates - Html файлы проекта основной файл index.html
4. routing.py - марруты сайта

## Схема взаимодействия модулей

![alt text](https://github.com/morgonxak/door/blob/master/rs/scheme.png)

  

