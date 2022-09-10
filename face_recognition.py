import cv2
import numpy as np
import os


data_path = 'data/'
face_data = []
labels = []
names = {}
class_id = 0

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

for f in os.listdir(data_path):
    if f.endswith('.npy'):
        names[class_id] = f[:-4]
        f_data = np.load(data_path + f)
        face_data.append(f_data)
        
        target = class_id * np.ones((f_data.shape[0],))
        class_id +=1 
        labels.append(target)
        
faces_data = np.concatenate(face_data, axis = 0)
faces_labels = np.concatenate(labels, axis = 0)

def dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(x, y, q_point, k = 5):
    vals = []
    n = x.shape[0]
    for i in range(n):
        dis = dist(q_point, x[i])
        vals.append((dis, y[i]))
    vals = sorted(vals)
    vals = vals[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:, 1], return_counts = True) # new_vals class is tuple of arrays
    index = new_vals[1].argmax()
    return new_vals[0][index]

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)    
        
    for face in faces:
        x, y, w, h = face
        offset = 10
        face_section = frame[y - offset : y + h + offset, x - offset: x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        
        prediction = knn(faces_data, faces_labels, face_section.flatten())
        #cv.putText(img, text, point, fontFace, fontScale, color, thickness, lineType)
        cv2.putText(frame, names[int(prediction)], (x + 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x +w, y + h), (0, 255, 255), 3)
        
    cv2.imshow('face', frame)    
        
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()     