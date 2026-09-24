import cv2

import numpy as np

import matplotlib.pyplot as plt

image = cv2.imread(
    r'C:\Users\ayush\OneDrive\Desktop\Documents\na.jpeg'
)

if image is None:
    print("Image not found!")
    exit()

print(type(image))

print(image.shape)

cv2.imshow('image', image)

cv2.waitKey(0)

image_resize = cv2.resize(image, (500, 500))

cv2.imshow('image_resize', image_resize)

cv2.waitKey(0)

image_flip = cv2.flip(image, 1)

cv2.imshow('Photos', image_flip)

cv2.waitKey(0)

image_crop = image[100:300, 200:500]

cv2.imshow('Photos', image_crop)

cv2.waitKey(0)


# =========================================================
# MORE FEATURES
# =========================================================

# 1. Rotate Image

image_rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Rotated Image', image_rotate)

cv2.waitKey(0)


# 2. Convert Image to Grayscale

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Image', image_gray)

cv2.waitKey(0)


# 3. Blur Image

image_blur = cv2.GaussianBlur(image, (15, 15), 0)

cv2.imshow('Blurred Image', image_blur)

cv2.waitKey(0)


# 4. Edge Detection

image_edges = cv2.Canny(image_gray, 100, 200)

cv2.imshow('Edge Detection', image_edges)

cv2.waitKey(0)


# 5. Increase Brightness

image_bright = cv2.convertScaleAbs(
    image,
    alpha=1.0,
    beta=50
)

cv2.imshow('Bright Image', image_bright)

cv2.waitKey(0)


# 6. Increase Contrast

image_contrast = cv2.convertScaleAbs(
    image,
    alpha=1.5,
    beta=0
)

cv2.imshow('Contrast Image', image_contrast)

cv2.waitKey(0)


# 7. Draw Rectangle

image_rectangle = image.copy()

cv2.rectangle(
    image_rectangle,
    (50, 50),
    (300, 250),
    (0, 255, 0),
    3
)

cv2.imshow('Rectangle', image_rectangle)

cv2.waitKey(0)


# 8. Draw Circle

image_circle = image.copy()

cv2.circle(
    image_circle,
    (250, 200),
    80,
    (255, 0, 0),
    3
)

cv2.imshow('Circle', image_circle)

cv2.waitKey(0)


# 9. Add Text

image_text = image.copy()

cv2.putText(
    image_text,
    'OpenCV Image Editing',
    (50, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow('Text', image_text)

cv2.waitKey(0)


# 10. Save Edited Image

cv2.imwrite(
    'edited_image.jpg',
    image_text
)

print("Edited image saved successfully!")


cv2.destroyAllWindows()