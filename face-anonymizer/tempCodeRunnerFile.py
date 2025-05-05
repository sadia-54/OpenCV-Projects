# pylint: disable=no-member
import cv2
import mediapipe as mp

# read image
img_path = 'data/face_img.jpeg'
img = cv2.imread(img_path)

cv2.imshow('img', img)
cv2.waitKey(0)

# detect faces

# blur faces

# save image
