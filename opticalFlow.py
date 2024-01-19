import cv2
import numpy as np
import time

def draw_hsv(flow):
    h,w = flow.shape[:2] 
    fx,fy = flow[:,:,0], flow[:,:,1] 
    
    ang = np.arctan2(fy,fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    
    hsv = np.zeros((h,w,3), np.uint8) #makes a "pixel array" of all black (3d array which can be interpreted as
    # h,w 2d array of 0 3-tuples)
    #HSV is hue saturation value
    hsv[...,0] = ang*(180 / (np.pi*2)) #rad to deg, so this determines the hue
    hsv[...,1] = 255 #full saturation
    hsv[...,2] = np.minimum(v*4, 255) # how bright the color is, so bright for moving fast and black for stationary.
    #for each pixel, 
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # makes into thing we can see
    
    return bgr

def draw_box(flow):
    fx,fy = flow[:,:,0], flow[:,:,1] 
    
    v = np.sqrt(fx*fx + fy*fy)
    
    topLeftx = 0
    topLefty = 0
    y = 0
    x = 0
    out = false
    while y < len(v):
        while x < len(v[0]):
            if v[y,x] > 50:
                topLeftx = x
                topLefty = y
                out = true
                break
            x += 10
        if out:
            break
        x = 0
        y += 10
    
    botRightx = 0
    botRighty = 0
    y = len(v)-1
    x = len(v[0])-1
    out = false
    while y > 0:
        while x > 0:
            if v[y, x] > 50:
                botRightx = x
                botRighty = y
                out = true
                break
            x -= 10
        if out:
            break
        x = len(v[0])-1
        y -= 10
    
    bgr = np.zeros((h,w,3), np.uint8)
    cv2.rectangle(bgr, (topLeftx,topLefty), (botRightx,botRighty), (0,255,0))
    
    return bgr

capture = cv2.VideoCapture(0) #should be webcam


has_frame, prev = capture.read()
gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    has_frame, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    start = time.time()
    
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_frame, None, .5, 3, 15, 2, 5, 1.2, 0)
    #flow = cv2.calcOpticalFlowPyrLK(gray_prev, gray_frame) may attempt sparce optical flow
    
    gray_prev = gray_frame #preparation for next frame/loop
    
    img = draw_hsv(flow) 
    #box = draw_box(flow) 
    
    end = time.time()
    deltaT = end - start
    fps = 1/deltaT
    print(fps)
    
    cv2.putText(img, 
    text = "FPS: " + str(fps), 
    org = (10,30), 
    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale = 1, 
    color = (255,255,255), 
    thickness = 1)
    
    #this is for viewing results, would most likely be removed.
    #cv2.imshow("Image", gray_frame)
    cv2.imshow("HSV", img)
    #cv2.imshow("BOX", box)
    key = cv2.waitKey(1)
    if key == 27:
        break