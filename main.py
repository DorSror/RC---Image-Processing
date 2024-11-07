import cv2 # 4.10.0
import time
import os
import numpy as np
import face_recognition as fr # requires cmake (manual install on windows), wheel, dlib

# settings constants (current is 16:9 360p, 30fps)
WIDTH = 640
HEIGHT = 360
FPS_CAP = 30
CAM_INDEX = 0 # usually 0, could be 1 if more than one camera is connected
WINDOWS_ORIGIN_X = 0
WINDOWS_ORIGIN_Y = 0
DISPLAY_NAME = "Webcam"
TRACKBAR_NAME = "Trackbars"
TRACKBAR_COUNT = 7 # change based on number of trackbars
TRACKBAR_HEIGHT = 36 # standard height for trackbar in trackbar menu window
TRACKBAR_DEF_VALUE = 0 
HSV_HUE_UPPER_BOUND = 179
HSV_DEF_UPPER_BOUND = 255
MIN_CONTOUR_AREA = 256
MAX_CONTOUR_AREA = WIDTH * HEIGHT / 2 
IMAGE_DIR = "ImageDir"
HAAR_FACE_CASCADE_PATH = "haar/haarcascade_frontalface_default.xml"
HAAR_EYE_CASCADE_PATH = "haar/haarcascade_eye_tree_eyeglasses.xml" # for considering glasses
HAAR_SMILE_CASCADE_PATH = "haar/haarcascade_smile.xml"

# define mouse click callback function
def onImageClick(event, xPosClick, yPosClick, flags, params):
    global onClickEvent, clickEventPoint
    onClickEvent = event
    clickEventPoint = (xPosClick, yPosClick)
    if onClickEvent == cv2.EVENT_LBUTTONDOWN: # left mouse button press
        print(f"Left click pressed (event {onClickEvent}) at {clickEventPoint}")
    if onClickEvent == cv2.EVENT_RBUTTONDOWN: # right mouse button press
        print(f"Right click pressed (event {onClickEvent}) at {clickEventPoint}")
    if onClickEvent in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}: # either mouse button release
        print(f"Mouse press ended (event {onClickEvent}) at {clickEventPoint}")

# define trackbar value change callback function(s)
def lowHueTrackbarCallback(value):
    global lowHue
    lowHue = value

def highHueTrackbarCallback(value):
    global highHue
    highHue = value

def lowSatTrackbarCallback(value):
    global lowSat
    lowSat = value

def highSatTrackbarCallback(value):
    global highSat
    highSat = value

def lowValTrackbarCallback(value):
    global lowVal
    lowVal = value

def highValTrackbarCallback(value):
    global highVal
    highVal = value

def blockRangeTrackbarCallback(value):
    global blockMask
    blockMask = value

# set image capture
cam_input = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cam_input.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam_input.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam_input.set(cv2.CAP_PROP_FPS, FPS_CAP)
cam_input.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

# face recognition
face_cascade = cv2.CascadeClassifier(HAAR_FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(HAAR_EYE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(HAAR_SMILE_CASCADE_PATH)

# walk through image directory
for path, dirs, files in os.walk(IMAGE_DIR):
    print(f"Working directory: {path}\nDirectories: {dirs}\nFolders: {files}")

# set windows and functionalities
cv2.namedWindow(DISPLAY_NAME) # create window for image display
cv2.setMouseCallback(DISPLAY_NAME, onImageClick)
cv2.namedWindow(TRACKBAR_NAME) # create window for trackbars
cv2.createTrackbar("lowHue", TRACKBAR_NAME, TRACKBAR_DEF_VALUE, HSV_HUE_UPPER_BOUND, lowHueTrackbarCallback)
cv2.createTrackbar("highHue", TRACKBAR_NAME, HSV_HUE_UPPER_BOUND, HSV_HUE_UPPER_BOUND, highHueTrackbarCallback)
cv2.createTrackbar("lowSat", TRACKBAR_NAME, TRACKBAR_DEF_VALUE, HSV_DEF_UPPER_BOUND, lowSatTrackbarCallback)
cv2.createTrackbar("highSat", TRACKBAR_NAME, HSV_DEF_UPPER_BOUND, HSV_DEF_UPPER_BOUND, highSatTrackbarCallback)
cv2.createTrackbar("lowVal", TRACKBAR_NAME, TRACKBAR_DEF_VALUE, HSV_DEF_UPPER_BOUND, lowValTrackbarCallback)
cv2.createTrackbar("highVal", TRACKBAR_NAME, HSV_DEF_UPPER_BOUND, HSV_DEF_UPPER_BOUND, highValTrackbarCallback)
cv2.createTrackbar("Mask Type", TRACKBAR_NAME, TRACKBAR_DEF_VALUE, 1, blockRangeTrackbarCallback)
cv2.resizeWindow(TRACKBAR_NAME, WIDTH // 2, TRACKBAR_HEIGHT * TRACKBAR_COUNT)

# move windows to starting position
cv2.moveWindow(DISPLAY_NAME, WINDOWS_ORIGIN_X, WINDOWS_ORIGIN_Y)
cv2.moveWindow(TRACKBAR_NAME, WINDOWS_ORIGIN_X, WINDOWS_ORIGIN_Y + HEIGHT)

# destroying unused windows without commenting out rest of code
#cv2.destroyWindow(TRACKBAR_NAME)

# exit if camera port failed to open
if not cam_input.isOpened():
    print("Failed opening the image input.")
    exit()

# parameters and global variables initialization
time_last = time.time() # time at start of execution
onClickEvent = -1 # undefined event (no event at start)
clickEventPoint = (0, 0) 
lowHue = TRACKBAR_DEF_VALUE
highHue = HSV_HUE_UPPER_BOUND
lowSat = TRACKBAR_DEF_VALUE
highSat = HSV_DEF_UPPER_BOUND
lowVal = TRACKBAR_DEF_VALUE
highVal = HSV_DEF_UPPER_BOUND
blockMask = 0 # 0 = false, 1 = true

# display images
while True:
    ret, frame = cam_input.read() # capture image
    if not ret: # check if image capture failed
        print("Failed image capture.")
        break

    if cv2.waitKey(1) & 0xff == 27: # Esc to close program
        break

    ## image processing
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # image in HSV
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscaled image

    # create lower and upper bounds for tracking objects using HSV color space
    # for each color you wish to track, you need to create new bounds and a range (can neglect sat, val)
    lower_bound = np.array([lowHue, lowSat, lowVal])
    upper_bound = np.array([highHue, highSat, highVal])

    # create a mask using the bounds
    mask_hsv = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # mask the original frame
    masked_frame = 0
    if blockMask: # invert the mask
        mask_hsv = cv2.bitwise_not(mask_hsv)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_hsv)

    # get countours from mask and display
    contours, hierarchy = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < MIN_CONTOUR_AREA or contour_area > MAX_CONTOUR_AREA:
            continue
        cv2.drawContours(frame, [contour], 0, (255, 0, 255), 2) # display contour in purple
        bounding_rect_x, bounding_rect_y, bounding_rect_w, bounding_rect_h = cv2.boundingRect(contour)
        # display contour's bounding rectangle
        cv2.rectangle(frame, 
                      (bounding_rect_x, bounding_rect_y), 
                      (bounding_rect_x + bounding_rect_w, bounding_rect_y + bounding_rect_h),
                      (0, 255, 0), 
                      3)
        
    # detect faces (+ eyes) and display bounding rectangles
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for face in faces:
        face_x, face_y, face_w, face_h = face
        cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 160, 160), 2)
        face_ROI = frame_gray[face_y:face_y + face_h, face_x:face_x + face_w] # isolate face in ROI
        # find eyes within face region of interest
        eyes = eye_cascade.detectMultiScale(face_ROI, 1.3, 5)
        for eye_x, eye_y, eye_w, eye_h in eyes:
            # adjust eyes coordinates to face in the original frame
            eye_x += face_x
            eye_y += face_y
            # display bounding rectangle for eyes in face
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (160, 160, 255), 1)
        # find smile within face region of interest
        smiles = smile_cascade.detectMultiScale(face_ROI, 1.5, 25)
        for smile_x, smile_y, smile_w, smile_h in smiles:
            # adjust eyes coordinates to face in the original frame
            smile_x += face_x
            smile_y += face_y
            # display bounding rectangle for eyes in face
            cv2.rectangle(frame, (smile_x, smile_y), (smile_x + smile_w, smile_y + smile_h), (0, 255, 160), 1)
        

    ## display FPS
    # calculate FPS
    current_time = time.time()
    moment_fps = 1 / (current_time - time_last)
    time_last = current_time

    # draw FPS
    fps_text = f"FPS: {min(int(moment_fps), FPS_CAP)}"
    cv2.putText(frame, fps_text, (WIDTH - 70, 14), cv2.FONT_HERSHEY_PLAIN, 1, (0, 220, 0), 2)

    # display image
    cv2.imshow(DISPLAY_NAME, frame)

# end execution
cam_input.release()
cv2.destroyAllWindows()
