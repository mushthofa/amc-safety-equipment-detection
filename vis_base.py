# 22-09-2018
# Azure Czure

# Preferably do not modify this file (has multiple copies)

# -- TOP SECRET --
project_id = "564cf32d-c93c-40d3-a20a-a5c66bcd5d48"
prediction_key = "8939846c14154971adef2c0c202d3126"


from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math

# Variables
import os
pc_name = os.environ['COMPUTERNAME']
if pc_name == 'RXPS':
    yolo_path = r"C:\Development\AM Challenge\YOLO\\"
else:
    yolo_path = r"..\YOLO\\"
classes_path = yolo_path + 'yolov3.txt'
weights_path = yolo_path + 'yolov3.weights'
config_path = yolo_path + 'yolov3.cfg'
scale = 0.0035
conf_threshold = 0.5
nms_threshold = 0.4
predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def print_all_status(img, person_list, helmet_list, goggles_list, gasdet_list):
    cv.putText(img, 'Helmet', (100, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.putText(img, 'Safety Goggles', (300, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.putText(img, 'Gas Detector', (500, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    person_list = np.sort(person_list)
    for idx, person in enumerate(person_list):
        helmet_ok = person in helmet_list
        goggles_ok = person in goggles_list
        gasdet_ok = person in gasdet_list
        print_status(img, idx, person, helmet_list, goggles_ok, gasdet_ok)

def print_status(img, idx, person, helmet_ok, goggles_ok, gasdet_ok):
    red = (0, 0, 255)
    green = (0, 255, 0)
    cv.putText(img, 'Person ' + str(person) + ': ', (0, 30 + 20*idx), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.putText(img, 'OK' if helmet_ok else 'NO', (100, 30 + 20*idx), cv.FONT_HERSHEY_SIMPLEX, 0.5, green if helmet_ok else red, 2)
    cv.putText(img, 'OK' if goggles_ok else 'NO',(300, 30 + 20*idx), cv.FONT_HERSHEY_SIMPLEX, 0.5, green if goggles_ok else red, 2)
    cv.putText(img, 'OK' if gasdet_ok else 'NO', (500, 30 + 20*idx), cv.FONT_HERSHEY_SIMPLEX, 0.5, green if gasdet_ok else red, 2)

# returns: dictionary of class / tag names to a LIST of:
# (x, y, w, h, conf, cid)
# conf = confidence / probability
# cid = class / tag ID (0-49)
def yolo_process_frame(img, net):
    Width = img.shape[1]
    Height = img.shape[0]
    class_ids = []
    confidences = []
    boxes = []
    res = dict()
    for c in classes:
        res[c] = []

    # Predict
    blob = cv.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        conf = confidences[i]
        cid = class_ids[i]
        res[classes[cid]].append((x, y, w, h, conf, cid))
        
    return res


# returns: dictionary of class / tag names to a LIST of:
# (x, y, w, h, conf, cid)
# conf = confidence / probability
# cid = class / tag ID (0-49)
def custom_process_frame_US(img, min_prob):
    # img is a numpy array, predictor accepts BufferedReader ..
    width = img.shape[1]
    height = img.shape[0]

    # file_like = BytesIO(img)
    # print(file_like)
    success, encoded_image = cv.imencode('.jpg', img)
    pdata = encoded_image.tobytes()
    pires = predictor.predict_image(project_id, pdata)
    res = dict()

    for p in pires.predictions:
        if p.probability >= min_prob:
            x = int(p.bounding_box.left * width)
            y = int(p.bounding_box.top * height)
            w = int(p.bounding_box.width * width)
            h = int(p.bounding_box.height * height)
            conf = p.probability
            cid = hash(p.tag_name) % 50
            if not(p.tag_name) in res:
                res[p.tag_name] = []
            res[p.tag_name].append((x, y, w, h, conf, cid))
    
    return res


def draw_box2(img, label, color, x, y, w, h):
    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

def draw_box(img, label, color, box):
    draw_box2(img, label, color, box[0], box[1], box[2], box[3])

def box_center(box):
    return (box[0] + box[2]/2, box[1] + box[3]/2)

def box_dist(box1, box2):
    center1 = box_center(box1)
    center2 = box_center(box2)
    return abs(center1[0] - center2[0]) + abs(center1[1]-center2[1])

def match_boxes(boxes1, boxes2, people_list):
    boxes2_available = [True for i in range(len(boxes2))]
    matches = dict()
    for idx1, box1 in enumerate(boxes1):
        min_dist = math.inf
        min_idx = None
        for idx2, box2 in enumerate(boxes2):
            if boxes2_available[idx2] == True:
                cur_dist = box_dist(box1, box2)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_idx = idx2
        if min_idx != None:
            matches[idx1] = min_idx
            boxes2_available[min_idx] = False
    
    if len(people_list) == 0:
        last_nr = 0
    else:
        last_nr = max(people_list) + 1
    result = []
    for idx1, box1 in enumerate(boxes1):
        if idx1 in matches:
            result.append(people_list[matches[idx1]])
        else:
            result.append(last_nr)
            last_nr = last_nr + 1
    return result
