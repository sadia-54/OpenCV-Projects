# pylint: disable=no-member
import cv2
import mediapipe as mp

# read image
img_path = 'data/face_img.jpeg'
img = cv2.imread(img_path)
img = cv2.resize(img, (800, 600))

H, W, _ = img.shape

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rbg)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

    cv2.imshow('img', img)
    cv2.waitKey(0)

# blur faces

# save image
