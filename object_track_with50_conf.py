from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLOv8 model (trained on persons)
model = YOLO("person.pt")

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture("vid.mp4")

# Get video properties for saving output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'XVID'
out = cv2.VideoWriter("output_tracking.mp4", fourcc, fps, (width, height))

# Class names
class_names = model.names

# Set confidence threshold
conf_threshold = 0.5  # <- Adjust this value as needed (e.g., 0.3, 0.5, 0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf < conf_threshold:
            continue  # Skip low confidence detections

        bbox = [x1, y1, x2 - x1, y2 - y1]  # Format: [x, y, w, h]
        detections.append((bbox, conf, cls_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cls_name = class_names[track.det_class] if hasattr(track, 'det_class') else 'object'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id} {cls_name}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show and save the frame
    cv2.imshow("YOLOv8 Object Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
