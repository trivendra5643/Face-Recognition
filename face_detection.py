import cv2

video = cv2.VideoCapture(0) #id of webcam - 0 for default
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 7) #Scaling Factor and No. of neighbors
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)#p1, p2, color, thickness
        
    cv2.imshow('video frame', frame)
        
    #Terminate on Keyboard
    key_pressed = cv2.waitKey(1) & 0xFF #255
    if key_pressed == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()   