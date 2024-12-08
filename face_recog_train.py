import cv2 # 4.10.0
import time
import os
import dill
import numpy as np
import yaml # pyyaml
import face_recognition as fr # requires cmake (manual install on windows), wheel, dlib.
# fr is computation heavy - preferably run this on GPU. (todo?)

# load relevant settings from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
FPS_CAP = config["FPS_CAP"]
CAM_INDEX = config["CAM_INDEX"]
WINDOWS_ORIGIN_X = config["WINDOWS_ORIGIN_X"]
WINDOWS_ORIGIN_Y = config["WINDOWS_ORIGIN_Y"]
DISPLAY_NAME = config["DISPLAY_NAME"]
FONT_DEF = config["FONT_DEF"]
FONT_FACE_DEF = config["FONT_FACE_DEF"]

# set image capture
cam_input = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cam_input.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam_input.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam_input.set(cv2.CAP_PROP_FPS, FPS_CAP)
cam_input.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

# exit if camera port failed to open
if not cam_input.isOpened():
    print("Failed opening the image input.")
    exit()

## define image processing behavior
# given a frame (captured_image) in RGB
def detect_faces(captured_image):
    faces = fr.face_locations(captured_image)
    return faces

# load face encodings from memory, detect new faces and add to memory
def add_new_faces(faces_encodings):
    # load face encodings from memory
    faces_encodings = [[], []] # first array is names, second array is encodings
    faces_encodings_new = [[], []]
    try:
        with open("faces_encodings.pkl", 'rb') as f:
            faces_encodings = dill.load(f)
    except FileNotFoundError:
        pass
    else:
        f.close()

    # find the new faces and add them to the known ones
    for encoding in faces_encodings:
        matchings = fr.compare_faces(faces_encodings[1], encoding)
        face_index = {faces_encodings.index(encoding) + 1}
        if True in matchings: # known face detected
            print(f"Face #{face_index} is already known. Proceeding to next face.")
        else: # new face
            new_face_name = input(f"Face #{face_index} is new! Please add a name: ")
            faces_encodings_new[0].append(new_face_name)
            faces_encodings_new[1].append(encoding)
    faces_encodings[0] += faces_encodings_new[0]
    faces_encodings[1] += faces_encodings_new[1]

    # save into memory
    with open("faces_encodings.pkl", 'wb') as f:
        dill.dump(faces_encodings, f)
        f.close()

# the main function when a user selects an image
def manage_fr_logic(captured_frame, faces):
    # display frame with bounding rectangles for each face
    for face in faces:
        top, right, bottom, left = face
        cv2.rectangle(captured_frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(captured_frame, f"Unknown #{faces.index(face) + 1}", (left, top - 5), 
                    FONT_FACE_DEF, 0.5, (0, 0, 255), 1)
    cv2.waitKey(1)
    cv2.imshow(DISPLAY_NAME, captured_frame)
    cv2.waitKey(1)
    user_choice = input(f"{len(faces)} face(s) detected. Do you wish to proceed? (Y/n) ").lower()
    while not user_choice in {'n', 'no', 'y', 'yes'}:
        user_choice = input(f"Unknown choice, please try again. (Y/n) ").lower()
    if user_choice in {'n', 'no'}:
        return
    else:
        faces_encodings = fr.face_encodings(captured_frame)
        add_new_faces(faces_encodings)
    
# parameters and global variables initialization
time_last = time.time() # time at start of execution

# display images
while True:
    ret, frame = cam_input.read() # capture image
    if not ret: # check if image capture failed
        print("Failed image capture.")
        break

    if cv2.waitKey(1) & 0xff == 27: # Esc to close program
        break

    # hold Space to capture images until faces are found
    if cv2.waitKey(1) & 0xff == 32:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(frame_rgb)
        if faces:
            manage_fr_logic(frame, faces)

    ## display FPS
    # calculate FPS
    current_time = time.time()
    moment_fps = 1 / (current_time - time_last)
    time_last = current_time

    # draw FPS
    fps_text = f"FPS: {min(int(moment_fps), FPS_CAP)}"
    cv2.putText(frame, fps_text, (WIDTH - 70, 14), FONT_DEF, 1, (0, 220, 0), 2)

    # display image
    cv2.imshow(DISPLAY_NAME, frame)

# end execution
cam_input.release()
cv2.destroyAllWindows()
