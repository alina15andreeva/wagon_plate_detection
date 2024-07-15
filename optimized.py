from ultralytics import YOLO
import cv2
from tqdm import tqdm
from datetime import datetime
import numpy as np
from datetime import timedelta

import tensorflow as tf
import matplotlib.patches as patches
import math

from statistics import mode

from collections import OrderedDict
import json

def detect(model, image):
    results = model(image, conf = 0.25, iou = 0.7, verbose=False)
    detections = []

    for r in results:
        result = json.loads(r.tojson())
        for box in result:
            position = box['box']
            keypoints = []
            if 'keypoints' in box:
                for i, keypoint_x in enumerate(box['keypoints']['x']):
                    keypoints.append({
                        'x': int(keypoint_x),
                        'y': int(box['keypoints']['y'][i]),
                        'visible': round(box['keypoints']['visible'][i], 2)
                    })
            box_dict = dict(OrderedDict([
                ('class', box['name']),
                ('confidence', round(box['confidence'], 2)),
                ('x', int(position['x1'])),
                ('y', int(position['y1'])),
                ('width', int(position['x2'] - position['x1'])),
                ('height', int(position['y2'] - position['y1'])),
                ('keypoints', keypoints),
                ('color', '#00ffcc')
            ]))
            detections.append(box_dict)

    return detections


def most_common(List):
    return mode(List)


def ocr_func(detected_object):
    dict = "0123456789                                                                                       "
    img = detected_object

    if img is None:
        return ''

    image_np = np.array(img)

    imgC, imgH, imgW = 3, 48, 320

    h, w = img.shape[:2]
    ratio = w / float(h)

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    elif imgW / 2.4 > math.ceil(imgH * ratio):
        resized_w = int(imgW / 2.4)
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')

    resized_image = resized_image / 255
    resized_image -= 0.5
    resized_image /= 0.5

    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, 0:resized_w, :] = resized_image

    image_input = padding_im.reshape(1, 48, 320, 3)

    interpreter.set_tensor(input_details[0]['index'], image_input)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    res = []
    sco = []
    for x in range(40):
        if np.argmax(output_data[0][x]) != 0:
            res.append(dict[np.argmax(output_data[0][x]) - 1])
            sco.append(output_data[0][x][np.argmax(output_data[0][x])])

    return ''.join(np.array(res)[sorted(sorted(range(len(sco)), key=lambda sub: sco[sub])[-8:])])


yolov8 = YOLO("/Users/alinaandreeva/Downloads/test_dir/best.pt")

video_path = 'test_video.MP4'
output_video_path = 'output1.mp4'

confidence_threshold = 0.7

video = cv2.VideoCapture(video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

detection_info = []
prev_time = None
prev_x_coordinate = None
train_count = 0
time_delta_test = None

img_dict = {}

model_path = "/Users/alinaandreeva/Downloads/TheosAI2/wagon-roi-main/Recognition/TFLite Model/wagon_num_rec_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    ret, frame = video.read()

    if not ret:
        break

    detections = detect(yolov8, frame)

    if detections:
        detections = [d for d in detections if d['confidence'] >= confidence_threshold]

        if detections:
            highest_confidence_detection = max(detections, key=lambda x: x['confidence'])

            x = int(highest_confidence_detection['x'])
            y = int(highest_confidence_detection['y'])
            width = int(highest_confidence_detection['width'])
            height = int(highest_confidence_detection['height'])

            if x >= 50:
                current_time = datetime.now()
                if prev_time is not None:
                    time_delta = current_time - prev_time
                else:
                    time_delta = None
                prev_time = current_time

                if prev_x_coordinate is not None:
                    x_coordinate_delta = abs(x - prev_x_coordinate)
                    x_coordinate_delta_direction = x - prev_x_coordinate
                else:
                    x_coordinate_delta = None
                prev_x_coordinate = x

                detected_object = frame[y:y + height, x:x + width]

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"Confidence: {highest_confidence_detection['confidence']:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                output_filename = f'detected_{int(video.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'

                detected_number = ocr_func(detected_object)

                img_dict[output_filename] = detected_number

                current_time = datetime.now()
                detection_confidence = highest_confidence_detection['confidence']
                detection_x = x
                screenshot_file_name = output_filename

                if time_delta != None:
                    if time_delta < timedelta(milliseconds=1000) and x_coordinate_delta < 20:
                        train_count = train_count
                    else:
                        train_count += 1

                detection_info.append({
                    'time': current_time,
                    'confidence': detection_confidence,
                    'x_coordinate': detection_x,
                    'screenshot_file_name': screenshot_file_name,
                    'time_delta': time_delta,
                    'x_coordinate_delta': x_coordinate_delta,
                    'train_count': train_count
                })

    output_video.write(frame)

    pbar.update(1)

pbar.close()

video.release()
output_video.release()

# Print the detection information
# for info in detection_info:
#    print(f"Time: {info['time']}, Confidence: {info['confidence']}, X Coordinate: {info['x_coordinate']}, Screenshot File Name: {info['screenshot_file_name']}, Time Delta: {info['time_delta']}, X Coordinate Delta: {info['x_coordinate_delta']}, Train Count: {info['train_count']}")

# print(f"Output video saved to {output_video_path}")


# Delete train_nums with the number of detections less than 5
detected_count = {}
for detected_train_num in detection_info:
    train_number = detected_train_num['train_count']
    if train_number not in detected_count:
        detected_count[train_number] = 1
    else:
        detected_count[train_number] += 1

max_train_count = max(info['train_count'] for info in detection_info)

filtered_detection_info = []

for y in detection_info:
    train_number = y.get('train_count')
    if train_number is not None:
        if train_number in detected_count and detected_count[train_number] >= 10:
            filtered_detection_info.append(y)

detection_info = filtered_detection_info


train_num = 0
num_list = []
num_dict = {}

max_train_count = max(info['train_count'] for info in detection_info)
while train_num <= max_train_count:
    for info in detection_info:
        if train_num == info['train_count']:
            for key, value in img_dict.items():
                if info['screenshot_file_name'] == key:
                    num_list.append(value)
    if len(num_list) != 0:
        num_dict[train_num] = most_common(num_list)
    num_list = []
    train_num += 1

x_dict = {}
train_num = 0
while train_num <= max_train_count:
    closest_coordinate = None
    closest_distance = float('inf')
    middle_coordinate = 0
    count = 0
    first_coordinate = None
    last_coordinate = None

    for info in detection_info:
        if train_num == info['train_count']:
            if count == 0:
                first_coordinate = info['x_coordinate']
                first_time = info['time']
                count = 1
            last_coordinate = info['x_coordinate']
            last_time = info['time']

    if count > 0 and first_coordinate is not None and last_coordinate is not None:
        middle_coordinate = (first_coordinate + last_coordinate) / 2

        for info in detection_info:
            if train_num == info['train_count']:
                distance = abs(info['x_coordinate'] - middle_coordinate)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_coordinate = info['x_coordinate']

        x_dict[train_num] = [last_coordinate - first_coordinate, closest_coordinate, first_time, last_time]

    train_num += 1

prev_info = None
i = 1

for train_n in num_dict:
    for train_x in x_dict:
        if train_x == train_n:
            current_info = (
            num_dict[train_n], 'left_to_right' if x_dict[train_x][0] > 0 else 'right_to_left', x_dict[train_x][1],
            x_dict[train_x][2], x_dict[train_x][3])

            if (prev_info is None or current_info != prev_info) and len(current_info[0]) == 8:
                print('Train number: ', i, 'Detected number: ', current_info[0], 'Train direction: ',
                      current_info[1], 'Middle coordinate: ', current_info[2], 'First Time: ', current_info[3],
                      'Last time: ', current_info[4])
                i += 1

            prev_info = current_info
