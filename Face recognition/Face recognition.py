import cv2

def detect(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.1, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
   # print(rects)


cap = cv2.VideoCapture(0)

while(True):
    what, img = cap.read()

    rects, img = detect(img)
    box(rects, img)

    # Display the resulting frame
    cv2.imshow('Face Recognition',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()