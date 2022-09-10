import cv2
dog = cv2.imread('dog.png')
gray_dog = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Dog Image', dog)
cv2.imshow('Gray Dog', gray_dog)
cv2.waitKey(0) #Infinite time wait - 0
cv2.destroyAllWindows()