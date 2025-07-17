from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('runs/detect/yolo11n_cucumber_detection2/weights/best.pt')

img = cv2.imread("C:/Users/y0808/Downloads/IMG_1675.jpg") # 検出したいキュウリの静止画像のパス
if img is not None:
    results = model.predict(source=img, conf=0.1, verbose=False) # confを低くして試す
    for r in results:
        annotated_img = r.plot()
        cv2.imshow("Static Image Test", annotated_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("静止画像が見つかりません。")