import cv2
import torch

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    return cap

def process_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results

def draw_detections(frame, results):
    for *box, conf, cls in results.xyxy[0]:
        label = f'{results.names[int(cls)]}: {conf:.2f}'
        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        frame = cv2.putText(frame, label, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    return frame

def main():
    model = load_model()
    cap = open_camera()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        results = process_frame(frame, model)
        frame = draw_detections(frame, results)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()