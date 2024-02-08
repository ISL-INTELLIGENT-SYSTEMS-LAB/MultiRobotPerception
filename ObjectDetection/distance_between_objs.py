import cv2
import cvzone
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

color_path = 'test_obj_vid.avi'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

model = YOLO('.//Yolo-Weights/yolov8n.pt')
class_names = ['person','bicycle', 'car', 'motorbike', 'aeroplane','bus', 'train', 'truck', 'boat', 'traffic ligth',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant','bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
               'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 
               'tooth brush']
min_distance = None

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame :
        continue
      
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    results = model(color_image, stream=True)
    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in range(len(boxes)):
            
            if int(boxes[box].cls[0]) == 41:
                #BoundingBox
                x1, y1, x2, y2 = boxes[box].xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
                w, h = x2-x1, y2-y1
                center_x = int((x1 + x2) / 2)
                center_y= int((y1 + y2) / 2)
                center_of_bbox = ((x2+x1)//2, (y2+y1)//2)
                distance_to_center =  depth_frame.get_distance(int((x2+x1)/2), int((y2+y1)/2))
                detected_objects.append((center_x, center_y, distance_to_center, x1, x2, y1, y2))
                
                cvzone.cornerRect(color_image, (x1, y1, w, h))
                
                radius = 6
                color = (0, 0, 255)  # Red color in BGR
                thickness = 1
                cv2.circle(color_image, center_of_bbox, radius, color, thickness)
                
                #Confidence
                conf = math.ceil((boxes[box].conf[0]*100))/100

                #Class
                cls = boxes[box].cls[0]
                
        #order the objects from left to right
        detected_objects.sort(key=lambda a: a[3])
        obj_ids = [i for i in range(1, len(detected_objects)+1)]
        
        for i in range(len(detected_objects)):
            for j in range(i+1, len(detected_objects)):
                obj1 = detected_objects[i]
                obj2 = detected_objects[j]
                
                if obj1[3] == obj2[3]:
                    if obj2[5] >= obj1[5]:
                        detected_objects.insert(i, obj2)
             
        for i in range(len(detected_objects)):
            for j in range(i+1, len(detected_objects)):
                obj1 = detected_objects[i]
                obj2 = detected_objects[j]
                
                y_midpoint = (obj2[1] + obj1[1]) // 2
                x_midpoint = (obj2[0] + obj1[0])// 2
                
                # Calculate real-world coordinates (using camera intrinsics)
                fx, fy, cx, cy = 612.5603637695312, 612.6820068359375, 321.65362548828125, 241.14561462402344
                
                x_real1 = (obj1[0] - cx) * obj1[2] / fx
                y_real1 = (obj1[1] - cy) * obj1[2] / fy
                x_real2 = (obj2[0] - cx) * obj2[2] / fx
                y_real2 = (obj2[1] - cy) * obj2[2] / fy
                
                #calculate the Euclidian distance
                distance_between_objs = math.sqrt((x_real1 - x_real2) ** 2 + (y_real1 - y_real2) ** 2 + (obj1[2] - obj2[2]) ** 2)
                
                if min_distance == None:
                    min_distance = distance_between_objs
                elif distance_between_objs < min_distance:
                    min_distance = distance_between_objs
                
                cv2.line(color_image, (obj1[0],obj1[1]), (obj2[0],obj2[1]), (255, 0, 0), 1)
                cv2.putText(color_image, f"{distance_between_objs*3.28084:.2f}ft",(x_midpoint, y_midpoint), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                

                try:
                    cv2.putText(color_image, f"{obj_ids[i]}", (obj1[4], obj1[5]), cv2.FONT_HERSHEY_TRIPLEX , 1,(255, 0, 0))
                    cv2.putText(color_image, f"{obj_ids[j]}", (obj2[4], obj2[5]), cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 0, 0))
                except IndexError:
                    print("Index out of range")

                                            
                #cvzone.putTextRect(color_image,f"{class_names[int(cls)]} {conf}", (max(0, x1), max(35,y1)), scale=0.6, thickness=1)
                #cvzone.putTextRect(color_image,f"Distance: {distance_to_center:.2f}m", (max(0, x1-25), max(35,y1-25)), scale=0.6, thickness=1)

            colorwriter.write(color_image)
    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)     
    cv2.imshow("Object Detection", color_image)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
    
        break