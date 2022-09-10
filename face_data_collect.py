import cv2
import numpy as np

#initiate camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0
face_data = []
data_path = 'data/'
file_name = input('Enter your name: ')

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue
    faces = sorted(faces, key = lambda f:f[2] * f[3])
    
    for face in faces[-1:]: #last face is largest
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        
        #crop face
        offset = 10
        face_section = frame[y - offset : y + h + offset, x - offset: x + w + offset] # default convention in frame, 1st y then x
        face_section = cv2.resize(face_section, (100, 100))
        
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        
    cv2.imshow('video frame', frame)    
    cv2.imshow('face section', face_section)    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
        
face_data = np.array(face_data)  
print(face_data.shape)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

np.save(data_path + file_name + '.npy', face_data)
print('Data successfully saved at ' + data_path + file_name + '.npy')


cap.release()
cv2.destroyAllWindows()