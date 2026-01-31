import cv2

# used this tutorial for reference: https://neptune.ai/blog/15-computer-visions-projects
# NOTE: go back and check why sometimes nostril detected as eye/ smiling mouth?

cap = cv2.VideoCapture(0)
cap.set(6,640)
cap.set(4, 420)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# detect eyes??
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get corners around face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    #draw box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #eyes
        eyes = eyeCascade.detectMultiScale(imgGray)
        #draw bounding box for eyes
        for (ex, ey, ew, eh) in eyes:
            img = cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')