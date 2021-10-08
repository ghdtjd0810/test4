# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:12:35 2021

@author: LG
"""

# cv2 모듈을 불러 온다. 
import numpy as np
import cv2
from matplotlib import pyplot as plt
'''
# 100번째 있는 사진의 샘플을 본다. 
img = cv2.imread("A220148XX_04547.jpg")
plt.imshow(img)
plt.show()
'''


image = cv2.imread("A220148XX_04547.jpg")/255
print(image.shape)


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