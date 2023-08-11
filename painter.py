import cv2
import time
import handtrackingmodule as htm
import numpy as np
import os

def paint():
	overlayList=[] 

	brushThickness = 10
	eraserThickness = 100

	# Purple colour
	drawColor=(255,0,255)


	xp, yp = 0, 0
	# Canvas page
	imgCanvas = np.zeros((480, 640, 3), np.uint8)

	# Image from Menu folder
	folderPath="Menu"
	myList=os.listdir(folderPath)

	# To show all image
	for imPath in myList:
	    image=cv2.imread(f'{folderPath}/{imPath}')
	    overlayList.append(image)

	menu=overlayList[0] 
	cap = cv2.VideoCapture(0)
	cap.set(3,640) # width
	cap.set(4,480) # height

	detector = htm.handDetector(detectionCon=0.50,maxHands=1)#making object

	success = True

	while success:

	    # Import image
	    success, img = cap.read()
	    img=cv2.flip(img,1) # Mirror View
	    
	    # Find Hand Landmarks
	    img = detector.findHands(img) 
	    lmList,bbox = detector.findPosition(img, draw=False)
	    
	    if len(lmList)!=0:
	        x1, y1 = lmList[8][1],lmList[8][2]    # tip of index finger
	        x2, y2 = lmList[12][1],lmList[12][2]  # tip of middle finger
	        
	        # Check which fingers are up
	        fingers = detector.fingersUp()

	        # Selecting portion
	        if fingers[1] and fingers[2]:
	            xp,yp=0,0
	            
	            if y1 < 63:
	                if 200 < x1 < 290: 
	                    menu = overlayList[0]
	                    drawColor = (255, 0, 255) # Purple Brush

	                elif 310 < x1 < 390:
	                    menu = overlayList[1]
	                    drawColor = (0, 0, 255)  # Red Brush

	                elif 410 < x1 < 490:
	                    menu = overlayList[2]
	                    drawColor = (0, 255, 0)  # Green Brush

	                # Eraser
	                elif 500 < x1 < 610:   
	                    menu = overlayList[3]
	                    drawColor = (0, 0, 0)

	            # Just a rectangle        
	            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


	        # If Drawing Mode - Index finger is up
	        if fingers[1] and fingers[2] == False:
	            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
	            # print("Drawing Mode")

	            if xp == 0 and yp == 0:
	                xp, yp = x1, y1 
	            
	            # Eraser fuctions
	            if drawColor == (0, 0, 0):
	                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
	                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
	            else:
	                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)#gonna draw lines from previous coodinates to new positions 
	                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
	            xp,yp=x1,y1  
	           
	# Merging all pages

	    # Converting img to gray
	    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
	    
	    # Converting into binary image and thn inverting - MASK
	    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
	    
	    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
	    
	    # Merging original img with imgInv 
	    img = cv2.bitwise_and(img,imgInv)

	    # cv2.imshow('imgInv', img)
	    
	    # Merging img and imgcanvas

	    img = cv2.bitwise_or(img,imgCanvas)


	    # Displaying menu image
	    img[0:63,0:640]= menu

	    cv2.imshow("Image", img)
	    # cv2.imshow("Canvas", imgCanvas)
	    # cv2.imshow("Inv", imgInv)
	    if cv2.waitKey(1) == 27:
	        success = False

	cv2.destroyAllWindows()
