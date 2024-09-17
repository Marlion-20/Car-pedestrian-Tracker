import cv2

# our video
video = cv2.VideoCapture("video 2.mp4")

# pre-trained car classifier
car_classifier_file = "car.xml"
pedestrian_Classifier_file = "haarcascade_fullbody.xml"

# our car & pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_Classifier_file)

while True:
    # read the current frame
    (read_successful, frame) = video.read()
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale((grayscale_frame))

    # draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # display the gray scale image
    cv2.imshow("Marlion AI", frame)

    # hold the window
    key = cv2.waitKey(1)

    # close program if Q is pressed
    if key==81 or key==113:
        break
video.release()
print("code completed")
