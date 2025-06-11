from ultralytics import YOLO
import cv2
import numpy as np

# 学習済みきゅうり検出モデルをロード
model = YOLO('runs/detect/yolov8n_cucumber_detection/weights/best.pt')

MM_PER_PIXEL = 0.5 # 仮の値。実際の環境で計測・調整が必要

# 画像で推論を実行
image_path = 'path/to/new_cucumber_image.jpg'
results = model.predict(source=image_path, conf=0.25)

# 結果の処理
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Image not found at {image_path}")
else:
    for r in results:
        boxes = r.boxes # Boxes object (xyxy, conf, cls)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # バウンディングボックスの座標
            conf = box.conf[0] # 信頼度
            cls = int(box.cls[0]) # クラスID

            # きゅうりの場合のみ処理
            if model.names[cls] == 'cucumber':
                # バウンディングボックスの高さから長さを概算
                height_pixels = y2 - y1
                estimated_length_mm = height_pixels * MM_PER_PIXEL

                # 結果を画像に描画
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{model.names[cls]}: {estimated_length_mm:.1f}mm (Conf: {conf:.2f})"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Detected {model.names[cls]} with length {estimated_length_mm:.1f}mm, Confidence: {conf:.2f}")

    # 結果画像を保存または表示
    cv2.imwrite('predicted_cucumber_with_length.jpg', img)
    # cv2.imshow('Cucumber Detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()