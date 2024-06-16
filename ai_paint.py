import cv2
import time
import os
import numpy as np 
import HandTrackingModule as htm 

folderPath = "header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
    
# print(len(overlayList))

header = overlayList[0]
drawcolor = (255, 0, 255)

#######################

brushThickness = 15
eraserThickness = 90

#######################

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) 
    if len(lmList) != 0:
        # print(lmList)
        
        # tip of index and middle fingure
        x1, y1 = lmList[8][1:]   
        x2, y2 = lmList[12][1:]   
        
        # 3. Check which finger is up
        fingers = detector.fingersUP()
        # print(fingers)
        
    
    
        # 4. If Selection mode -Two fingers are up 
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            #checking for the click
            if y1 < 100:
                if 300 < x1< 490:
                    header = overlayList[2]
                    drawcolor = (0, 0, 222)
                elif 550< x1 < 750:
                    header = overlayList[1]
                    drawcolor = (38, 178, 28)
                elif 800< x1 < 950:
                    header = overlayList[3]
                    drawcolor = (55, 234, 244)
                elif 1050< x1 < 1200:
                    header = overlayList[0]
                    drawcolor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawcolor, cv2.FILLED)
                    
            
        # 5. If Drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawcolor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                
            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushThickness)
                
            xp, yp = x1, y1
            
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
        
    # setting the header image
    img[0:100,0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Canvas",imgCanvas)
    cv2.imshow("Image",img)
    cv2.waitKey(1)