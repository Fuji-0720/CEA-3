from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# モデルの学習
results = model.train(
    data='cucumber_dataset.yaml',
    epochs=100,                  # 学習エポック数
    imgsz=640,                   # 画像のサイズ（モデルの入力サイズに合わせる）
    batch=16,                    # バッチサイズ
    name='yolov8n_cucumber_detection'
)