from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# モデルの学習
results = model.train(
    data='cucumber_dataset.yaml',
    epochs=50,                  # 学習エポック数
    imgsz=640,                   # 画像のサイズ（モデルの入力サイズに合わせる）
    batch=16,                    # バッチサイズ
    name='yolo11n_cucumber_detection',
)