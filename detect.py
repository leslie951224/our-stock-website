import cv2
from ultralytics import YOLO

# 1. 載入模型
model = YOLO('yolov8n.pt')

# 2. 開啟攝像頭
cap = cv2.VideoCapture(0)

print("正在啟動鏡像辨識系統... 按下 'q' 鍵可結束。")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # --- 關鍵步驟：將影像水平翻轉（鏡像） ---
        # 參數 1 表示水平翻轉，0 表示垂直翻轉，-1 表示兩者皆有
        frame = cv2.flip(frame, 1)

        # 3. 進行辨識 (使用翻轉後的畫面)
        results = model.predict(frame, conf=0.5, show=False)

        # 4. 繪製結果
        annotated_frame = results[0].plot()

        # 5. 顯示視窗
        cv2.imshow("YOLOv8 Mirror Reflection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()