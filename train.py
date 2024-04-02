# ## training
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# results = model.train(data='config.yaml', epochs=2, imgsz=640, device='cpu')

# validation
from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
model = YOLO("/Users/subhamswaruppradhan/Desktop/capston/runs/detect/train10/weights/best.pt")

validation_results = model.val(data='config.yaml', imgsz=640, batch=16, conf=0.25, iou=0.6, device='cpu')
