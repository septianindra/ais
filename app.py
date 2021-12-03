from flask import Flask, render_template, Response, request
import numpy as np
import jyserver.Flask as jsf
import cv2
import datetime
import time
import os
import sys


app = Flask(__name__)

global capture
capture = 0

try:
    os.mkdir('./shots')
except OSError as error:
    pass


@jsf.use(app)
class App:
    def __init__(self):
        self.count = 0


camera = cv2.VideoCapture(0)
target = ''


def gen_frames():  # generate frame by frame from camera
    global capture

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    while True:
        classes = None
        with open('./model/object-detection-opencv/yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        # setup the network
        net = cv2.dnn.readNet('./model/object-detection-opencv/yolov3.weights',
                              './model/object-detection-opencv/yolov3.cfg')
        # Capture frame-by-frame
        # set width
        camera.set(3, 416)
        # set height
        camera.set(4, 416)
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            Width = frame.shape[1]
            Height = frame.shape[0]
            scale = 0.00392
            blob = cv2.dnn.blobFromImage(
                frame, scale, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, conf_threshold, nms_threshold)

            for i in indices:
                i = i
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(frame, class_ids[i], confidences[i], round(
                    x), round(y), round(x + w), round(y + h))

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 200)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            class_list = [i for i in indices]
            # num_persons = str(class_list.count(1))
            # title = 'person counter'
            # cv2.putText(frame, 'num_persons: ' + num_persons,
            #             bottomLeftCornerOfText,
            #             font,
            #             fontScale,
            #             fontColor,
            #             lineType)
            target = [i for i in indices]
            ret, buffer = cv2.imencode('.jpg', frame)
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/')
def index():
    return App.render(render_template('index.html', target=target))


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
