from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
model = YOLO("/Users/subhamswaruppradhan/Desktop/capston/runs/detect/train10/weights/best.pt")

# Run batched inference on a list of images
results = model('cap.jpg', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk