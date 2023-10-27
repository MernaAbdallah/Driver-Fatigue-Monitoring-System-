from flask import Flask, Response, render_template, redirect, url_for, request
# Required to run Yolov8 Model
import cv2

# video_detection is function which perform object detection on input video
from YOLO_Object_Detection import video_detection

app = Flask(__name__)  # we initalizing flask 

app.config['object_detection'] = 'Amr_Abdelaty'

# Now we will display output video with detection
def generate_frames(path_x = ''): # generate_frames : Take path of input video file and gives us the output with bounding boxes, labels and conf score  around each object detected
    
    yolo_output = video_detection(path_x)

    for detection in yolo_output:
        ref, buffer = cv2.imencode('.jpg',detection) # we are encoding the detection , any flask app requires encoded image to be converted into bytes
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n') # we used Content_type to convert individual frames to be replaced by subsequent frames
        
@app.route('/')
def home():
    return ('''
<div class="Drive Drowsiness Detction" 
     style="background-color:#F5F5DC; 
     color:black; 
     padding: 10px; 
     margin: 10px; 
     font-size: 100%; 
     border-radius: 10px; 
     box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);">
  <h1><center>Drive Drowsiness Detection</center></h1>   
  </div>''')


@app.route('/video')
def video():
    return Response(generate_frames(path_x='NITYMED_2.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

