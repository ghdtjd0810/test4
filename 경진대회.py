# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:12:35 2021

@author: LG
"""


import json
def load_json(fname):
    with open(fname, encoding = "utf 8") as f:
        json_obj =  json.load(f)
        
    return json_obj

#a=load_json('A220148XX_04547.json')

import pandas as pd

#a1 = pd.DataFrame(a)

b= load_json('A220148XX_04547.json')
c= load_json('A220148XX_04548.json')
d= load_json('A220148XX_04549.json')
b2 = pd.DataFrame(b)
c2 = pd.DataFrame(c)
d2 = pd.DataFrame(d)


# In[2]

# In[2]



# cv2 모듈을 불러 온다. 


import numpy as np
import cv2
from matplotlib import pyplot as plt

# 100번째 있는 사진의 샘플을 본다. 
img = cv2.imread("A220148XX_04547.jpg")
#plt.imshow(img)
#plt.show()



image = cv2.imread("A220148XX_04547.jpg")/255
print(image.shape)

image_50x50 = cv2.resize(image, (50, 50))
plt.imshow(image_50x50)
plt.show()

'''

data_height = 150
data_width = 150
channel_n = 3
images = np.zeros((100, data_height, data_width, channel_n)) 

for n, path in enumerate("A220148XX_04547.jpg"):
    image = cv2.imread("A220148XX_04547.jpg")
    image = cv2.resize(image, (data_height, data_width))
    images[n, :, :, :] =image
    
# In[1]
print(images.shape)
'''
# In[1]
import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# In[Loading image]
img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# In[Detecting objects]
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# In[Showing informations on the screen]
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

