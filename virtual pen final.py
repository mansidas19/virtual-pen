#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time


# In[2]:


import cv2


# In[3]:


import numpy as np


# In[6]:


#STEP 1: FINDING COLOR RANGE OF TARGET PEN AND SAVE IT

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
#set size of window
cap.set(3,1280)
cap.set(4,720)

cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the L-lower and U-upper range of 
# H-hue,S-saturation and V-value channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
 
while True:
    #ret stores status and frame stores the image captured
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the new values of the trackbar in real time as the user changes them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
    # Set the lower and upper HSV range according to the value selected from trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filter the image and get the binary mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
 
    # visualize the real part of the target color
    # parameters src1 ,src2 and destination i.e what we finally want out of whole
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so we can stack with others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,frame,res))
    
    # Show this stacked frame at 40% of the size.
    # showing all 3 frames i.e actual, hsv in gray and hsv in actual color of pen
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
    #ESC key= exit program
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # If the user presses `s` then save this array
    if key == ord('s'):
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        np.save('penval',thearray)
        break
    
cap.release()
cv2.destroyAllWindows()


# In[11]:


#STEP 2:GETTING RID OF EXTRA NOISE IN BACKGROUND AND FOCUSING ON MAIN OBJECT

# This variable determines if we want to load color range from memory  
load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
#set size of window
cap.set(3,1280)
cap.set(4,720)

# Creating A 5x5 kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

while(1):
    #ret stores status and frame stores the image captured
    ret, frame = cap.read()
    if not ret:
        break
      
    # to flip the frame by 180 degree so that we get the object in same direction like mirror image 
    # because we saved the array in the previous step like that
    frame = cv2.flip( frame, 1 )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
    else:             
        lower_range  = np.array([26,80,147])
        upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise.
    mask = cv2.erode(mask,kernel,iterations = 1)
    # erode keeps those pixels that have all values 1 else it is removed just like and operation
    mask = cv2.dilate(mask,kernel,iterations = 2)
    # dilate is like or operation if atleast one value is 1 it is not removed
    
    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack all frames and show it
    stacked = np.hstack((mask_3,frame,res))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
    # run till the escape key is pressed , value for escape key is 27
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


# In[13]:


#STEP 3: TRACKING THE TARGET PEN

load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

# set the window to auto-size so we can view this full screen.
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This threshold is used to filter noise
noiseth = 500

while(1):
    
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
    else:             
            lower_range = np.array([0,0,0])
            upper_range = np.array([0,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours in the frame.
    # parameters of function are source image, countour retieval method ,countour approximation method
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also make sure its size is bigger than noise threshold. 
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
        
        # Grab the biggest contour with respect to area
        c = max(contours, key = cv2.contourArea)
        
        # Get bounding box coordinates around that contour
        x,y,w,h = cv2.boundingRect(c)
        
        # Draw that bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)        

    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


# In[14]:


#STEP 4:FINALLY WRITING WITH VIRTUAL PEN

load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

# Initializing the canvas on which we will draw upon
canvas = None

x1,y1=0,0

# Threshold for noise
noiseth = 800

while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initialize the canvas as a black image of the same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
    else:             
            lower_range  = np.array([26,80,147])
            upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also its size is bigger than the noise threshold.
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
        # This is true when we writing for the first time or when writing again when the pen had disappeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the canvas
            # parameters the array on which we have to draw, coordinates of point 1 and point 2 which isto be connected
            # colour of line and thickness of line
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Merge the canvas and the frame.
    frame = cv2.add(frame,canvas)
    
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None

cv2.destroyAllWindows()
cap.release()


# In[9]:





# In[ ]:




