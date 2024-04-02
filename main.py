import cv2
from ultralytics import YOLO
import argparse
import supervision as sv
import os
import subprocess
def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def play_mp3(mp3_file_path):
    try:
        subprocess.run(['afplay', mp3_file_path])
    except FileNotFoundError:
        print("Error: 'afplay' command not found. Make sure you are running on macOS.")

mp3_file_path = '/Users/subhamswaruppradhan/Downloads/alarm.mp3'


def main():
    args = parse_argument()
    frame_width , frame_height = args.webcam_resolution


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model = YOLO("yolov8n.pt")
    model = YOLO('yolov8n.yaml').load('yolov8n.pt') # build from YAML and transfer weights


    # .load("/Users/subhamswaruppradhan/Desktop/capston/runs/detect/train10/weights/best.pt")
    # model2 = YOLO("/Users/subhamswaruppradhan/Desktop/capston/runs/detect/train10/weights/best.pt")
   
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    alarm_triggered = False
    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        print(result)
        # break
        detections = sv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        print(labels)

        for label in labels:
            if "cell phone" in label.lower() or "knife" in label.lower():
                play_mp3(mp3_file_path)


        cv2.imshow("yolov8", frame)
   
        if(cv2.waitKey(30) == 27):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()