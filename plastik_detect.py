import torch
import cv2

# Load model YOLOv5 custom (ganti dengan path model kamu)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestpla.pt')

# Buka kamera laptop (0 = kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Jalankan deteksi
    results = model(frame)

    # Ambil hasil deteksi dan gambar bounding box di frame
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        conf_text = f"{conf:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Tampilkan frame hasil deteksi
    cv2.imshow('Plastic Detection (YOLOv5)', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup jendela
cap.release()
cv2.destroyAllWindows()

