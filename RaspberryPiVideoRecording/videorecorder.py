import cv2
import threading
from datetime import date

# dd_mm_YY
date = date.today().strftime("%d_%m_%Y")

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print(f"Starting {self.previewName}")
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    size = (int(cam.get(3)), int(cam.get(4)))
    filename = f"video_{camID}_{date}.avi"
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        video.write(frame)
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cam.release()
    video.release()
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", 1)
thread2 = camThread("Camera 2", 2)
thread1.start()
thread2.start()
