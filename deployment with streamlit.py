# from ultralytics import YOLO
# import cv2
#
# model = YOLO('C:/Users/asus/Downloads/Last_Notebook_with_50_and_100_epochs/Models/best_with_50_epochs.pt')
#
# model.predict(source = '0', show = True, conf = 0.5) # source = 0 it will use webcam ,
# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Define the path to your YOLOv8 model
model_path = 'C:/Users/asus/Downloads/Last_Notebook_with_50_and_100_epochs/Models/best_with_50_epochs.pt'

# Load the YOLO model
model = YOLO(model_path)

# Function to display tracker options
def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = "data.yaml"  # Use your custom YAML file name
        return is_display_tracker, tracker_type
    return is_display_tracker, None

# Function to display detected frames
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

# Set page layout and title
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Driver Drowsiness Detection using YOLOv8")

# Sidebar for ML model configuration
st.sidebar.header("DL Model Config")

# Confidence slider
confidence = float(st.sidebar.slider("Select Confidence of YOLOv8 model", 25, 100, 40)) / 100

# Function to play webcam and detect objects
def play_webcam(conf, model):
    # Open the webcam source
    vid_cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

    st_frame = st.empty()
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            _display_detected_frames(conf, model, st_frame, image)
        else:
            vid_cap.release()
            break

# Button to start object detection
if st.sidebar.button('Detect Objects'):
    play_webcam(confidence, model)
