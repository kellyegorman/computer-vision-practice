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
        
        # detect if eyes are looking left/right/up/down/center??
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            radius = int(round((ew + eh) * 0.25))
            img = cv2.circle(img, eye_center, radius, (255, 0, 255), 2)
            direction = ""

            # FIX ME: always says center??
            if eye_center[0] < ex + ew // 3:
                direction = "Left"
            elif eye_center[0] > ex + 2 * ew // 3:
                direction = "Right"
            else:
                direction = "Center"

            # write above the eye box what direction the eye is looking
            cv2.putText(img, direction, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.imshow('face_detect', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')