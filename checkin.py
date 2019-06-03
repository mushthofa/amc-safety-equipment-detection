# 22-09-2018
# Azure Czure
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models
from azure.cognitiveservices.vision.customvision.prediction.models.prediction import Prediction
import cv2
import time
import numpy as np

# import imageio

USE_CAMERA = True

classFile = 'yolov3.txt'
weightFile = r"C:\Development\AM Challenge\YOLO\yolov3.weights"
cfgFile = 'yolov3.cfg'
test_img_url = 'sample.jpg'
video_url = '00011.mp4'
# vid_out_url = '/Users/apple/Downloads/00280_output.mp4'
net = cv2.dnn.readNet(weightFile, cfgFile)
# constants
scale = 0.0035
conf_threshold = 0.5
nms_threshold = 0.4
N_cache = 5

# Now there is a trained endpoint that can be used to make a prediction
# -- PLEASE CHANGE THESE KEYS IF YOU ARE NOT BASILE --
project_id = "564cf32d-c93c-40d3-a20a-a5c66bcd5d48"
prediction_key = "8939846c14154971adef2c0c202d3126"
predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

global pass_flag, pass_Counters, boxsize_Tracker

# Read classes file
with open(classFile, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# wanted classes
global targetClasses  # Change this according to work scenario
targetClasses = ['person', 'helmet', 'safetygoggles', 'gasdetector', 'glove']  # Change this according to work scenario
ctime = ''  # Checkin time
pass_flag = False
# generate random color per-class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


class box:
    def __init__(self):
        self.left = 0.0
        self.right = 0.0
        self.width = 0.0
        self.height = 0.0


class YoloResult:
    def __init__(self):
        self.bounding_box = box
        self.tag_name = ''
        self.probability = 0


def init_pass_CNT(targetClasses):
    pass_CNT = {}
    for target in targetClasses:
        pass_CNT[target] = 0
    return pass_CNT


def init_box_tracker(targetClasses):
    box_tracker = {}
    for target in targetClasses:
        box_tracker[target] = 0
    return box_tracker


def intersects(box1, box2):
    return not ((box1.left + box1.width) < box2.left or box1.left > (box2.left + box2.width) or box1.top < (
                box2.top + box2.height) or (box1.top + box1.height) > box2.top)


def draw_prediction(img, label, confidence, x, y, x_plus_w, y_plus_h):
    color = COLORS[0]
    cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)
    cv2.putText(img, label, (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('frame', img)
    print("\t" + label + ": {0:.2f}%".format(confidence))


def display_results(predictions, frame, width, height):
    for prediction in predictions:
        if prediction.probability > conf_threshold:
            left = prediction.bounding_box.left * width
            top = prediction.bounding_box.top * height
            right = left + prediction.bounding_box.width * width
            down = top + prediction.bounding_box.height * height
            draw_prediction(frame, prediction.tag_name, prediction.probability * 100, left, top, right, down)


def checkin_logic(results, img, targetClasses):
    global pass_flag

    y = 10
    tags = []
    for result in results:
        if result.probability > conf_threshold:
            tags.append(result.tag_name)
            if USE_CAMERA:
                if result.tag_name in targetClasses:
                    boxsize_Tracker[result.tag_name] = boxsize_Tracker[
                                                           result.tag_name] + result.bounding_box.width * result.bounding_box.height

    for target in targetClasses:
        if target in tags:
            pass_Counters[target] = pass_Counters[target] + 1
            cv2.putText(img, target + ' ' + str(pass_Counters[target]), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        else:
            if pass_Counters[target] == 0:
                cv2.putText(img, target + ' Not Detected', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(img, target + ' ' + str(pass_Counters[target]), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
        y = y + 20
        if USE_CAMERA:
            if pass_Counters[target] != 0:
                boxsize_Tracker[target] = boxsize_Tracker[target] / pass_Counters[target]
    pass_flag = True
    for flag in targetClasses:
        if pass_Counters[flag] < N_cache:
            pass_flag = False
    if pass_flag and USE_CAMERA:
        global ctime
        if ctime == '':
            ctime = time.ctime()
        cv2.putText(img, 'You are all set! Checked in at time: ' + ctime, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def YoloProcessFrame(img, net):
    Height, Width, ch = img.shape
    confidences = []
    persons = []
    # Predict
    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    myYoloResults_list = []
    myYoloResult = YoloResult()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] in targetClasses:
                center_x = detection[0]
                center_y = detection[1]
                w = detection[2]
                h = detection[3]
                x = center_x - w / 2
                y = center_y - h / 2
                myYoloResult.tag_name = str(classes[class_id])
                myYoloResult.probability = float(confidence)
                myYoloResult.bounding_box.left = x
                myYoloResult.bounding_box.top = y
                myYoloResult.bounding_box.width = w
                myYoloResult.bounding_box.height = h
                myYoloResults_list.append(myYoloResult)
    return myYoloResults_list


def set_targetClasses(argin):
    global targetClasses
    targetClasses = argin


pass_Counters = init_pass_CNT(targetClasses)
boxsize_Tracker = init_box_tracker(targetClasses)


def detect_targets():
    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
        skip_frame = 0
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            skip_frame = skip_frame + 1
            if skip_frame % 10 != 0:
                continue

            # Our operations on the frame come here
            height, width, channels = frame.shape

            success, encoded_frame = cv2.imencode('.jpg', frame)
            img_data = encoded_frame.tobytes()
            results = predictor.predict_image(project_id, img_data, None)
            myAzureResults_list = results.predictions
            myYoloResults_list = YoloProcessFrame(frame, net)

            # Person Logics: Use yolo person prediction but others use Azure
            for i in range(0, len(myAzureResults_list) - 1):
                predict_azure = myAzureResults_list[i]
                if predict_azure.tag_name == 'person':
                    for yolo_result in myYoloResults_list:
                        if yolo_result.tag_name == 'person':
                            myAzureResults_list[i].probability = yolo_result.probability
                            myAzureResults_list[i].bounding_box.left = yolo_result.bounding_box.left
                            myAzureResults_list[i].bounding_box.top = yolo_result.bounding_box.top
                            myAzureResults_list[i].bounding_box.width = yolo_result.bounding_box.width
                            myAzureResults_list[i].bounding_box.height = yolo_result.bounding_box.height

            checkin_logic(myAzureResults_list, frame, targetClasses)
            # Display the results.
            display_results(myAzureResults_list, frame, width, height)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cap = cv2.VideoCapture(video_url)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        skip_frame = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            skip_frame = skip_frame + 1
            if skip_frame % 10 != 0:
                continue

            if ret == True:
                # Display the resulting frame
                height, width, channels = frame.shape

                myYoloResults_list = YoloProcessFrame(frame, net)

                success, encoded_frame = cv2.imencode('.jpg', frame)
                img_data = encoded_frame.tobytes()
                results = predictor.predict_image(project_id, img_data, None)
                myAzureResults_list = results.predictions
                myYoloResults_list = YoloProcessFrame(frame, net)

                # Person Logics: Use yolo person prediction but others use Azure
                for i in range(0, len(myAzureResults_list) - 1):
                    predict_azure = myAzureResults_list[i]
                    if predict_azure.tag_name == 'person':
                        for yolo_result in myYoloResults_list:
                            if yolo_result.tag_name == 'person':
                                myAzureResults_list[i].probability = yolo_result.probability
                                myAzureResults_list[i].bounding_box.left = yolo_result.bounding_box.left
                                myAzureResults_list[i].bounding_box.top = yolo_result.bounding_box.top
                                myAzureResults_list[i].bounding_box.width = yolo_result.bounding_box.width
                                myAzureResults_list[i].bounding_box.height = yolo_result.bounding_box.height

                checkin_logic(myAzureResults_list, frame, targetClasses)
                # Display the results.
                display_results(myAzureResults_list, frame, width, height)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    # When everything done, release the captureqq
    cap.release()
    cv2.destroyAllWindows()
    return pass_flag
