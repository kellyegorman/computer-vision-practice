import cv2
import face_recognition

imgmain = face_recognition.load_image_file('example_images/charlie.jpg')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('example_images/kayla.jpg') 

#catch index out of range error (no faces)
try: 
    faceLoc = face_recognition.face_locations(imgmain)[0]
except IndexError:
    print("No face detected in the image.")
    exit()

# detect second face
try: 
    faceLoc2 = face_recognition.face_locations(imgmain)[1]
    # print(faceLoc2)
except IndexError:
    print("Only one face.")

encodeElon = face_recognition.face_encodings(imgmain)[0]
cv2.rectangle(imgmain, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
# draw box around second face (if there is one)
encodeElon2 = None
if 'faceLoc2' in locals():
    encodeElon2 = face_recognition.face_encodings(imgmain)[1]
    cv2.rectangle(imgmain, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElon) 
faceDis = face_recognition.face_distance([encodeElon], encodeElon)  
print(results, faceDis)
cv2.putText(imgmain, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Image Main', imgmain)
cv2.waitKey(0)
# cv2.destroyAllWindows()
