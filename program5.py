import cv2
from ultralytics import YOLO

# --- 1. 学習済みモデルのロード ---
# ここには学習によって作成された best.pt ファイルの正しいパスを入力してください。
model = YOLO('runs/detect/yolo11n_cucumber_detection/weights/best.pt')

# --- 2. キャリブレーション値の設定 ---
# !!!!! 重要 !!!!!
# ここに、カメラキャリブレーションによって得られた正確な値を入力してください。
# これが「1ピクセルあたり何mm」に相当するかの値です。
# 初期段階では仮の数値でも動きますが、正確な長さを得るためには必須のステップです。
MM_PER_PIXEL = 0.5  # 例: 1ピクセルが0.5ミリメートルに相当すると仮定

# --- 3. Webカメラの初期化 ---
# 0 は通常、デフォルトのWebカメラを指します。
# 複数のカメラがある場合や、特定のカメラを使用する場合は、数値を変えてみてください (例: 1, 2 など)。
cap = cv2.VideoCapture(1)

# カメラが正しく開けたかを確認
if not cap.isOpened():
    print("エラー: Webカメラを開けませんでした。カメラが接続されているか、または他のアプリケーションで使用されていないか確認してください。")
    exit()

print("Webカメラからのリアルタイム検出を開始します。'q' キーを押すと終了します。")

# --- 4. リアルタイム処理ループ ---
while True:
    # フレームを1枚読み込む
    ret, frame = cap.read()

    # フレームが正しく読み込めなかった場合
    if not ret:
        print("エラー: フレームを読み込めませんでした。Webカメラからの入力が停止した可能性があります。")
        break

    # YOLOモデルで推論を実行
    # verbose=False にすると、推論ごとの詳細なコンソール出力が抑制され、よりスムーズに動作します。
    # conf=0.5 は、信頼度（確信度）が50%以上の検出結果のみを表示する閾値です。調整可能です。
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 検出結果の処理と描画
    for r in results:
        # 検出されたバウンディングボックスの座標、信頼度、クラスIDを取得
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]形式の座標
        confs = r.boxes.conf.cpu().numpy()  # 信頼度
        clss = r.boxes.cls.cpu().numpy()    # クラスID

        # 各検出されたオブジェクトについてループ
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = confs[i]
            cls_id = int(clss[i])

            # クラスIDからクラス名を取得 (model.names はデータセットのクラス名リスト)
            class_name = model.names[cls_id]

            # 今回は「cucumber」クラスのみを対象
            if class_name == 'cucumber':
                # バウンディングボックスの高さ（ピクセル単位）を計算
                height_pixels = y2 - y1

                # 長さ判別ロジックの適用
                # !!!!! ここがあなたのMM_PER_PIXEL値が活きる場所です !!!!!
                estimated_length_mm = height_pixels * MM_PER_PIXEL

                # 結果をフレームに描画
                # バウンディングボックスの描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 緑色の矩形

                # テキストの描画 (クラス名、推定長さ、信頼度)
                text = f"{class_name}: {estimated_length_mm:.1f}mm (Conf: {conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # テキストは矩形の上に表示

    # 処理結果のフレームをウィンドウに表示
    cv2.imshow("Cucumber Length Detection (Press 'q' to quit)", frame)

    # 'q' キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. リソースの解放 ---
# カメラデバイスを解放
cap.release()
# 表示ウィンドウを全て閉じる
cv2.destroyAllWindows()

print("リアルタイム検出を終了しました。")