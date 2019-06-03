# 22-09-2018
# Azure Czure

# -- CHANGE THIS --
test_path = "00280.mp4"
output_path = "output280b.avi"
require_helmet = True
require_goggles = True
require_gasdet = True


from vis_base import *
import cv2

COLORS = np.random.uniform(0, 255, size=(100, 3))
win_title = "Arcelor Mittal Challenge 2018 - AzureCZure: Dynamic safety equipment detection (press q to quit)"

net = cv2.dnn.readNet(weights_path, config_path)
vidCap = cv2.VideoCapture(test_path)

nr_frames = vidCap.get(cv2.CAP_PROP_FRAME_COUNT)
vid_width = vidCap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if (output_path is not None):
    vidOut = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 15, (round(vid_width), round(vid_height)))
    print('Saving to file ' + output_path + '...')
else:
    vidOut = None


old_boxes = []
people_list = []
helmet_list = []
safetygoggles_list = []
gasdet_list = []
i = 0
while vidCap.isOpened() and i < nr_frames:
    r, img = vidCap.read()
    i = i + 1
    if i % 10 != 0 and vidOut is None:
        continue
    if i % 2 != 0 and vidOut is not None:
        continue
    if(img is None):
        break
    if i%50 == 0 and vidOut is not None:
        print('Frames ' + str(i) + '/'+ str(nr_frames))
    height, width = img.shape[:2]
    width = img.shape[1]
    height = img.shape[0]
    
    yolo_boxes = yolo_process_frame(img, net)['person']
    people_list = match_boxes(yolo_boxes, old_boxes, people_list)
    old_boxes = yolo_boxes

    # Remove cache for people out of the frame
    helmet_list = [person for person in helmet_list if person in people_list]
    safetygoggles_list = [person for person in safetygoggles_list if person in people_list]
    gasdet_list = [person for person in gasdet_list if person in people_list]
    
    # Loop over persons
    for idx, box in enumerate(yolo_boxes):

        # Extract box around person (with tolerance)
        x = max(box[0] - 10, 0)
        y = max(box[1] - 30, 0)
        w = min(box[2] + 20, width - x)
        h = min(box[3] - 20, height - y)
        cid = box[5]

        # Apply custom vision on this person
        cropped = img[y:y+h, x:x+w]
        cust_boxes = custom_process_frame_US(cropped, 0.3)
        if require_helmet and 'helmet' in cust_boxes and people_list[idx] not in helmet_list:
            helmet_list.append(people_list[idx])
        if require_goggles and 'safetygoggles' in cust_boxes and people_list[idx] not in safetygoggles_list:
            safetygoggles_list.append(people_list[idx])
        if require_gasdet and 'gasdetector' in cust_boxes and people_list[idx] not in gasdet_list:
            gasdet_list.append(people_list[idx])
        
        # Draw boxes
        draw_box(img, 'person ' + str(people_list[idx]), COLORS[cid], box) # actual person
        draw_box2(img, '', (128, 128, 128), x, y, w, h) # sent to custom vision (= gray)
        tags = ['helmet', 'safetygoggles', 'gasdetector']
        for t in tags:
            if t in cust_boxes:
                for box in cust_boxes[t]:
                    px, py, pw, ph = box[0:4]
                    px = px + x
                    py = py + y
                    cid = box[5]
                    draw_box2(img, t[0], COLORS[cid + 50], px, py, pw, ph)
        
    print_all_status(img, people_list, helmet_list, safetygoggles_list, gasdet_list)
    
    if(vidOut is not None):
        vidOut.write(img)
    else:
        cv2.imshow(win_title, img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord(' '):
            while(True):
                key = cv2.waitKey(0)
                if key & 0xFF == ord(' '):
                    break

vidCap.release()
if(vidOut is not None):
    print('Done.')
    vidOut.release()

