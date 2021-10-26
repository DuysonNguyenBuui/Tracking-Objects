import cv2
import numpy as np

# track first frame
tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture(0)
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
# Initialize tracker with  frame and bounding box
tracker.init(img, bbox)

def drawBox(img, bbox):
    x, y, h, w = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

while True:
    # start time
    timer = cv2.getTickCount()
    success, img = cap.read()

    # cap nhat gia tri trinh theo doi
    success, bbox = tracker.update(img)
    print(bbox)
    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    cv2.imshow("Track", img)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break
