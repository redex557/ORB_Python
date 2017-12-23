import cv2
import time
import sys
import os

# set path to videos
videoDir = "D:/Master/HTW-Aalen/E-Motion_Rennteam/Driverless/fabian_fischer/Testdaten/30fps"


# for webcam and video file
def cv2getVideoFrame():
    return cap.read()


# image or video
def cv2readDir(directory):
    try:
        file_list = os.listdir(path=directory)
    except FileNotFoundError:
        print("FileNotFoundError")
        sys.exit(-1)

    file_list.sort()
    return file_list


# Initiate ORB detector
# orb = cv2.ORB_create()  # default
# orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)  # default values
orb = cv2.ORB_create(nfeatures=200, scaleFactor=1.5, nlevels=8, edgeThreshold=100, firstLevel=0, WTA_K=2,
                     scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=60)  # optimiert

print("press n for next video")

while True:  # endless loop
    fileList = cv2readDir(videoDir)

    for file in fileList:
        print("File: ", file)
        path = os.path.join(os.getcwd(), videoDir, file)
        cap = cv2.VideoCapture(path)

        if cap is None:
            continue

        startTimeVideo = time.perf_counter()
        ret = True
        while ret:
            startTimeFrame = time.perf_counter()
            ret, img = cv2getVideoFrame()

            if ret:
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                # find the keypoints with ORB
                kp = orb.detect(img, None)

                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)

                # draw only keypoints location, not size and orientation
                img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

                cv2.imshow('Video', img2)
                if cv2.waitKey(1) & 0xFF == ord('n'):  # next video
                    break

            else:
                break

            endTimeFrame = time.perf_counter()
            #print("Frame: time for detection (in fractional seconds):", endTimeFrame - startTimeFrame)


        endTimeVideo = time.perf_counter()
        print("Video: time for detection (in fractional seconds):", endTimeVideo - startTimeVideo)


cap.release()
