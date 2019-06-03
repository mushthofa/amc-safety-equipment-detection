# 23-09-2019
# Azure Czure

# Linearly animates a forbidden zone from 1 to 2

from vis_base import *
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import telegram_client

COLORS = np.random.uniform(0, 255, size=(100, 3))
win_title = "Forbidden zone tracking (press q to quit)"
no_message_send = True

net = cv2.dnn.readNet(weights_path, config_path)
vidCap = cv2.VideoCapture(test_path)
length = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
i = 0

# Skip as needed
for j in range(0, skip_cnt):
    r, img = vidCap.read()
print("Begin...")

while True:
    r, img = vidCap.read()
    i = i + 1
    if i % frame_step != 0:
        continue

    img = cv2.resize(img, (640, 360))
    if rotate_right:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    width = img.shape[1]
    height = img.shape[0]

    # Call OD
    yolo_boxes = yolo_process_frame(img, net)
    if len(yolo_boxes[cname]) == 0:
        # No relevant activity
        continue

    # Create polygon
    pos = i / length
    npos = 1.0 - pos
    cur_fz = []
    for quad in zip(fz_poly1, fz_poly2):
        x1 = quad[0][0]
        y1 = quad[0][1]
        x2 = quad[1][0]
        y2 = quad[1][1]
        cur_fz.append((x1*npos+x2*pos, y1*npos+y2*pos))
    polygon = Polygon(cur_fz)

    # Draw forbidden zone
    poly_up = [[width * x[0], height * x[1]] for x in cur_fz]
    poly_up.append(poly_up[0])
    pts = np.array(poly_up, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, (0, 128, 255), 6)

    # Process and draw class
    violated = False
    for box in yolo_boxes[cname]:
        cid = box[5]
        feet = Point((box[0] + box[2] / 1.5) / width,
                    (box[1] + box[3]) / height)
        if polygon.contains(feet):
            violated = True
            draw_box(img, 'OH NO!', COLORS[cid + 1], box)
        else:
            draw_box(img, 'yolo', COLORS[cid], box)

    # Red boundary
    if violated:
        if no_message_send:
            no_message_send = False
            telegram_client.setLEDstatus(True)
            telegram_client.sendUpdateToBot("Get out of there!", user="Jeroen")
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), 60)

    cv2.imshow(win_title, img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
