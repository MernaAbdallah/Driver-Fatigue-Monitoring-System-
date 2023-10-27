from ultralytics import YOLO
import cv2
import math 
import pygame  
import time

# Initialize pygame for playing sound
pygame.init()

# Load the sound file (replace 'alarm_sound.wav' with the path to your sound file)
alarm_sound = pygame.mixer.Sound('alarm-sound.mp3')

def video_detection(path_x):
    video_capture = path_x
    # start video
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # out = cv2.VideoWriter('NITYMED_23.avi', cv2.VideoWriter_fourcc('M', 'J' , 'P' , 'G'), 10, (frame_width, frame_height)) # i want to save the output video with the detection / frame rate = 10 / fw , fh

    # model
    model = YOLO('C:/Users/amrAb/Downloads/New_Samsung/Deep_Learning_Project/Last_Notebook_with_50_and_100_epochs/Models/best_with_100_epochs.pt')

    # object classes
    classNames = ['awake', 'calling', 'chatting', 'closed_eye', 'drink', 'drowsy', 'eating', 'no_yawn', 'open_eye', 'smoking', 'yawn']

    # Variables for detecting the specific class continuously
    detection_duration = 0  # Duration for which the specific class has been detected continuously
    previous_class = None  # Previous detected class
    start_time = None  # Start time when the specific class is first detected

    while True:
        success, img = cap.read() # now we are reading a video frame by frame
        #stream = True ==> will use the generator and it is more efficient than normal
        results = model(img, stream=True) # then i pass this frame to model to scan it and make detection
        
        # here we are looking through each of the individual bounding boxes coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0] # output in this line is tensor 
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # put box in cam // we create bounding box around each of the detected objects
                cv2.rectangle(img, (x1, y1), (x2, y2), 	(255,228,181), 3) # image, start point, end point, color, thickness

                # confidence // box.conf[0] == we have the conf value in form in tensor so we use .ceil to convert it to integer
                confidence = math.ceil((box.conf[0] * 100))/100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                class_name = classNames[cls]
                label = f'{class_name}{confidence}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255,228,181], -1, cv2.FONT_HERSHEY_SIMPLEX)  # filled
                cv2.putText(img, label, (x1, y1-2), 0, 1, [85,107,47], thickness=2,lineType=cv2.FONT_HERSHEY_SIMPLEX)

                if class_name in (classNames[1],classNames[2],classNames[4],classNames[5],classNames[6]):
                    if class_name != previous_class:
                        start_time = time.time()  # Update start time when the specific class is first detected
                        print(f'start time : {start_time}')
                    else:
                        duration = time.time() - start_time  # Calculate the duration for which the specific class has been detected continuously
                        print(duration)
                        print(f'duration time : {duration}')
                        if duration >= 2:
                            alarm_sound.play()
                            detection_duration = duration
                            break
                    previous_class = class_name
                else:
                    previous_class = None
                    start_time = None

        yield img   # we will get individualframes with bounding boxes around a detected object with label and confidence score

#     out.write(img) # Here we saving the object detections in out in the file name 'NITYMED_22.mp4'
#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# out.release()
cv2.destroyAllWindows()