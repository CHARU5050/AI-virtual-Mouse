import cv2
import numpy as np
import handtrackingmodule as htm
import time
import autopy

wCam,hCam=640,480
frameR=100
smoothening=7

plocx,plocy=0,0
clocx,clocy=0,0


cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

ptime=0
ctime=0
detector=htm.handDetector(maxhands=1)
wscr,hscr=autopy.screen.size()
print(wscr,hscr)

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist,bbox=detector.findposition(img)

    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        fingers=detector.fingerUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(x1, (frameR, wCam-frameR),(0,wscr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hscr))
            clocx=plocx+(x3-plocx)/smoothening
            clocy = plocy + (y3 - plocy) / smoothening

            autopy.mouse.move(wscr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy=clocx,clocy

        if fingers[1] == 1 and fingers[2] == 1:
            length,img,lineinfo=detector.findDistance(8,12,img)
            print(length)
            if length<25:
                cv2.circle(img, (lineinfo[4],lineinfo[5]), 15, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click()





    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0,0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
