import cv2
import urllib
import pdb
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


class VideoCamera(object):
    def __init__(self):
        self.video = urllib.urlopen(
            'http://192.168.10.12:8080/video')  # cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        def __del__(self):
            self.video.release()

        def get_frame(self):
            bytes = ''
            while True:
                bytes += self.video.read(1024)
                a = bytes.find('xffxd8')
                b = bytes.find('xffxd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    img = cv2.imdecode(np.fromstring(
                        jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # pdb.set_trace()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]
                        ret, jpeg = cv2.imencode('.jpg', img)
                        return jpeg.tobytes()
