import datetime
import time
import cv2
import imutils

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)/2)
frame_height = int(cap.get(4)/2)
firstFrame = None

while True:
    ret, frame = cap.read()
    text = "Unoccupied"
    # decrease_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    decrease_frame = cv2.resize(frame,(frame_width, frame_height))
    gray = cv2.cvtColor(decrease_frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(frame, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    firstFrame = gray
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.cvtColor(thresh,cv2.COLOR_RGB2GRAY)

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        # x *= 4
        # y *= 4
        # w *= 4
        # h *= 4
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()