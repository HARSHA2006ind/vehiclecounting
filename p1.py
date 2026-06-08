import cv2
import csv
import collections
import numpy as np
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Detection confidence threshold
confThreshold = 0.1
nmsThreshold = 0.2

# Middle cross line position
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# Store COCO Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

print(classNames)
print(len(classNames))

# Vehicle classes
required_class_index = [2, 3, 5, 7]

# YOLO files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3.weights"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(
    modelConfiguration,
    modelWeights
)

# Random colors
np.random.seed(42)
colors = np.random.randint(
    0,
    255,
    size=(len(classNames), 3),
    dtype='uint8'
)

print("YOLO Loaded Successfully!")

# Open Video
cap = cv2.VideoCapture("video.mp4")

while True:

    success, img = cap.read()

    if not success:
        print("Video Finished")
        break

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    ih, iw, channels = img.shape

    # Draw counting lines
    cv2.line(
        img,
        (0, middle_line_position),
        (iw, middle_line_position),
        (255, 0, 255),
        2
    )

    cv2.line(
        img,
        (0, up_line_position),
        (iw, up_line_position),
        (0, 0, 255),
        2
    )

    cv2.line(
        img,
        (0, down_line_position),
        (iw, down_line_position),
        (0, 0, 255),
        2
    )

    # YOLO Processing
    input_size = 320

    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255,
        (input_size, input_size),
        [0, 0, 0],
        1,
        crop=False
    )

    net.setInput(blob)

    layersNames = net.getLayerNames()

    outputNames = [
        layersNames[i - 1]
        for i in net.getUnconnectedOutLayers()
    ]

    outputs = net.forward(outputNames)

    print("Objects detected:", len(outputs))

    cv2.imshow("Vehicle Detection", img)

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()