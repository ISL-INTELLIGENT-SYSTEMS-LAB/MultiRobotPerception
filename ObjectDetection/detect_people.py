import cv2
import cvzone
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Define the path for the output video file
color_path = 'test_obj_vid.avi'

# Initialize a VideoWriter object to write the output video
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

# Load the YOLO model
model = YOLO('.//Yolo-Weights/yolov8n.pt')

# Define the class names for the YOLO model
class_names = ['person','bicycle', 'car', 'motorbike', 'aeroplane','bus', 'train', 'truck', 'boat', 'traffic ligth',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant','bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
               'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 
               'tooth brush']

# Start an infinite loop to continuously capture frames from the RealSense pipeline, 
# run the YOLO model on each frame, and process the results
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue
    
    # Convert the color frame data to a numpy array
    color_image = np.asanyarray(color_frame.get_data())
    
    # Run the model on the color image
    results = model(color_image, stream=True)
    
    # Iterate over the results
    for r in results:
        boxes = r.boxes
        # Iterate over the boxes in each result
        for box in boxes:
            
            # Check if the class is a person
            if int(box.cls[0]) == 0:
            # Extract the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
                w, h = x2-x1, y2-y1
                
                # Draw a rectangle around the detected object
                cvzone.cornerRect(color_image, (x1, y1, w, h),rt=2, colorR=(0, 0, 255))
                
                #Confidence
                conf = math.ceil((box.conf[0]*100))/100

                #Class
                cls = box.cls[0]
            
                # Put text on the image indicating the class and confidence of the detection
                cvzone.putTextRect(color_image,f"{class_names[int(cls)]} {conf}", (max(0, x1), max(35,y1)), colorR=(0, 0, 255))

            # Write the color image to a file
            colorwriter.write(color_image)
     
    # Display the image with the detections       
    cv2.imshow("Object Detection", color_image)
    
    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
    
        break