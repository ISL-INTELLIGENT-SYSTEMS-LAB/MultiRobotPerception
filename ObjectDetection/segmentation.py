import cv2
import pyrealsense2 as rs
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import random

# COCO class labels
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define a color for each class
COLORS = {
    'person': [0, 255, 0],  # Green
    'dog': [255, 0, 0],  # Red
    'cup': [0, 0, 255],  # Blue
    # Add more colors for other classes
    # ...
    'toothbrush': [255, 255, 0]  # Yellow
}


def display_mask(image, prediction):
    masks = prediction[0]['masks'].detach().cpu().numpy()
    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        mask = np.where(mask >= 0.5, 255, 0)
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    colorwriter.write(image)

# Start the RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Prepare to save video stream
color_path = 'test_segmentation_vid.avi'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

# Specify the Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Stream the video until "q" is pressed
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame :
        continue
      
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert the image into a torch.Tensor
    color_image_tensor = torchvision.transforms.functional.to_tensor(color_image).unsqueeze(0)

    # Perform instance segmentation
    with torch.no_grad():
        prediction = model(color_image_tensor)
    
    # Filter predictions for the 'person', 'dog', and 'cup' classes
    indices = [i for i, label in enumerate(prediction[0]['labels']) if COCO_CLASSES[label] in ['person', 'dog', 'cup']]
    filtered_predictions = {k: v[indices] for k, v in prediction[0].items()}
        
    # Get the class names of the detected objects
    labels = filtered_predictions['labels'].cpu().numpy()
    class_names = [COCO_CLASSES[i] for i in labels]
    
    # Get the masks of the detected objects
    masks = filtered_predictions['masks'].cpu().numpy()

    # Get the bounding boxes of the detected objects
    boxes = filtered_predictions['boxes'].cpu().numpy().astype(int)
    
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image back to BGR
    #gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Create a new gray image
    gray_image = np.full_like(color_image, 128)

    # Change the color of the pixels of each detected object
    '''for i in range(masks.shape[0]):
        class_name = class_names[i]
        if class_name in COLORS:
            color_image[masks[i, 0] > 0.5] = COLORS[class_name]
        else:
            color_image[masks[i, 0] > 0.5] = [255, 255, 255]  # Default color'''
            
    # Change the color of the pixels of each detected object
    for i in range(masks.shape[0]):
        class_name = class_names[i]
        if class_name in COLORS:
            gray_image[masks[i, 0] > 0.5] = COLORS[class_name]
        else:
            gray_image[masks[i, 0] > 0.5] = [255, 255, 255]  # Default color
    
    # Draw the class name on each detected object
    for box, class_name in zip(boxes, class_names):
        cv2.putText(gray_image, class_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    print("Detected objects:", class_names)


    # Display the segmented image
    # Note: You need to implement the display_mask function to visualize the masks
    display_mask(gray_image, prediction)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()