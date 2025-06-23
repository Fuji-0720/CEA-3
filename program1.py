from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.predict(source='path/to/your/image.jpg', conf=0.25)

for r in results:
    print(r.boxes)
    r.save(filename='predicted_image.jpg')