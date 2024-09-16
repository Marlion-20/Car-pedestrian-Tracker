import cv2

# our image
image = "pedestrian 1.png"

# pre-trained car classifier
classifier_file = "car.xml"

# read image into open cv
img= cv2.imread(image)

#convert to grayscale
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create the car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)
print(cars)

# draw a rectangle around the car
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0), 3)

# show the image or display it
cv2.imshow("Marlion AI",img)

# hold the window
cv2.waitKey()

print("code completed")
