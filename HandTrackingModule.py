import cv2 
import mediapipe as mp
import time 

class handDetector():
    def __init__(self):
        # self.mode = mode 
        # self.maxHands = maxHands
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # in this mpHands.HAND_CONNECTION is for joining dots    
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id, lm)
                h,w,c=img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                self.lmList.append([id, cx, cy])
                # if id==4:
                if draw:
                    cv2.circle(img, (cx,cy),6,(255,0,255),cv2.FILLED)          
        return self.lmList
    
    def fingersUP(self):
        fingers = []
        
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        
        # 4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers             

def main():
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)    
    
    detector = handDetector()

    while True: 
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
    
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN , 3, (255,0,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1) 
    

if __name__ == "__main__":
    main()
    

"""   
# if just you want to access the webcam then we have code below:
import cv2 
import mediapipe as mp
import time 
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1) 
"""