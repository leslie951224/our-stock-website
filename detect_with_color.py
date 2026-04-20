import cv2
import numpy as np
from ultralytics import YOLO

# 載入模型
model = YOLO('yolov8n.pt')

def get_color_name(hsv_frame, box):
    # 取得物體中心點的顏色
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # 取得中心點像素的 HSV 值
    pixel_hsv = hsv_frame[cy, cx]
    h_value = pixel_hsv[0]
    s_value = pixel_hsv[1]
    v_value = pixel_hsv[2]

    # 簡單的顏色判斷邏輯 (HSV 範圍)
    if s_value < 40: return "White/Gray"
    if v_value < 40: return "Black"
    
    if h_value < 10 or h_value > 170: return "Red"
    elif h_value < 22: return "Orange"
    elif h_value < 33: return "Yellow"
    elif h_value < 78: return "Green"
    elif h_value < 131: return "Blue"
    elif h_value < 150: return "Violet"
    else: return "Red"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 轉換為 HSV 顏色空間供分析使用
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 執行 YOLO 辨識
    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # 取得座標
            b = box.xyxy[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # 辨識顏色
            color_name = get_color_name(hsv_frame, b)
            
            # 畫出方框與文字 (包含名稱與顏色)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_text = f"{label} ({color_name})"
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()