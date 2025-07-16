import cv2
from ultralytics import YOLO

# --- 1. 学習済みモデルのロード ---
# ここには学習によって作成された best.pt ファイルの正しいパスを入力してください。
# 例: 'runs/detect/yolov8_cucumber_detector_first_run/weights/best.pt'
# もしあなたがYOLOv11を使っている場合は、それに合わせてパスを修正してください。
model_path = 'runs/detect/yolo11n_cucumber_detection/weights/best.pt' # ここをあなたのパスに修正！
model = YOLO(model_path)

# --- 2. Webカメラの初期化 ---
# 0 は通常、デフォルトのWebカメラを指します。
# 複数のカメラがある場合や、特定のカメラを使用する場合は、数値を変えてみてください (例: 1, 2 など)。
cap = cv2.VideoCapture(1)

# カメラが正しく開けたかを確認
if not cap.isOpened():
    print("エラー: Webカメラを開けませんでした。カメラが接続されているか、または他のアプリケーションで使用されていないか確認してください。")
    exit()

print("Webカメラからのリアルタイムきゅうり検出を開始します。'q' キーを押すと終了します。")

# --- 3. リアルタイム処理ループ ---
while True:
    # フレームを1枚読み込む
    ret, frame = cap.read()

    # フレームが正しく読み込めなかった場合
    if not ret:
        print("エラー: フレームを読み込めませんでした。Webカメラからの入力が停止した可能性があります。")
        break

    # YOLOモデルで推論を実行
    # source=frame: 現在のフレームを推論の入力とする
    # conf=0.5: 信頼度（確信度）が50%以上の検出結果のみを表示する閾値。調整可能。
    # verbose=False: 推論ごとの詳細なコンソール出力を抑制し、よりスムーズに動作させる。
    results = model.predict(source=frame, conf=0.1, verbose=False)

    # 検出結果の処理と描画
    # ultralyticsライブラリの便利な機能 r.plot() を使用すると、
    # 検出されたバウンディングボックスとラベルを自動的に画像に描画してくれます。
    # これにより、手動で矩形やテキストを描画するコードを書く手間が省けます。
    for r in results:
        # r.plot() は検出結果を描画した画像をNumPy配列として返します。
        # これを直接 cv2.imshow に渡すことができます。
        annotated_frame = r.plot()
        
        # 検出されたきゅうりの情報もコンソールに表示（オプション）
        for box in r.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]
            # 今回は 'cucumber' クラスのみなので、特にフィルタリングは不要かもしれませんが、
            # 複数のクラスを検出するモデルの場合は、ここで class_name == 'cucumber' でフィルタリングできます。
            if class_name == 'cucumber':
                 print(f"検出: {class_name}, 信頼度: {confidence:.2f}")


    # 処理結果のフレームをウィンドウに表示
    cv2.imshow("Cucumber Detection (Press 'q' to quit)", annotated_frame)

    # 'q' キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. リソースの解放 ---
# カメラデバイスを解放
cap.release()
# 表示ウィンドウを全て閉じる
cv2.destroyAllWindows()

print("リアルタイムきゅうり検出を終了しました。")