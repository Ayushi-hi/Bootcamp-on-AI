import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r'c:\Users\ayush\OneDrive\Desktop\Documents\na.jpeg')
print(type(image))
print(image.shape)

cv2.imshow('image',image)
cv2.waitKey(0)
image_resize = cv2.resize(image,(500,500))
cv2.imshow('image_resize',image_resize)
cv2.waitKey(0)

image_flip = cv2.flip(image,1)
cv2.imshow('Photos',image_flip)
cv2.waitKey(0)

image_crop = image[100:300,200:500]
cv2.imshow('Photos',image_crop)
cv2.waitKey(0)
cv2.destroyAllWindows

