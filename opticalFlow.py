import cv2
import numpy as np
import time

def draw_hsv(flow):
    h,w = flow.shape[:2]
    fx,fy = flow[:,:,0], flow[:,:,1]
    
    ang = np.arctan2(fy,fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    
    hsv = np.zeros((h,w,3), np.uint8)
    hsv[...,0] = ang*(180 / (np.pi*2)) #rad to deg
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr

capture = cv2.VideoCapture(0) #should be webcam


has_frame, prev = capture.read()
gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    has_frame, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    start = time.time()
    
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_frame, None, .5, 3, 15, 3, 5, 1.2, 0)
    
    gray_prev = gray_frame #preparation for next frame/loop
    
    
    
    
    
    
    
    end = time.time()
    deltaT = end - start
    fps = 1/deltaT
    print(fps)
    
    cv2.putText(frame, 
    text = "FPS: " + str(fps), 
    org = (10,30), 
    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale = 1, 
    color = (0,0,0), 
    thickness = 1)
    
    #this is for viewing results, would most likely be removed.
    #cv2.imshow("Image", gray_frame)
    cv2.imshow("HSV", draw_hsv(flow))
    key = cv2.waitKey(1)
    if key == 27:
        break