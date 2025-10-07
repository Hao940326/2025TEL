import cv2, numpy as np, time

# ====== 參數 ======
CAM_INDEX = 1                     # 攝影機編號
CAP_WIDTH, CAP_HEIGHT = 1280, 720 # 想要的畫面解析度
HSV_LOW  = (0, 0, 200)            # 白色下界
HSV_HIGH = (179, 60, 255)         # 白色上界
KERNEL   = (5, 5)                 # 形態學濾波核
CIRC_MIN = 0.78                   # 圓形度下限
AREA_MIN_FRAC = 0.0003            # 面積下限（越小越寬鬆）
AREA_MAX_FRAC = 0.15              # 面積上限
R_MIN_PX = 10                     # 半徑下限
R_MAX_FRAC = 0.25                 # 半徑上限

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)

    # 嘗試設定攝影機輸出解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ 無法讀取攝影機影像")
            break

        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---- 抓白色 ----
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        kernel = np.ones(KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ---- 找輪廓 ----
        min_area = AREA_MIN_FRAC * H * W
        max_area = AREA_MAX_FRAC * H * W
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4.0 * np.pi * area / (peri * peri)
            if circ < CIRC_MIN:
                continue
            (xc, yc), r = cv2.minEnclosingCircle(c)
            if r < R_MIN_PX or r > min(H, W) * R_MAX_FRAC:
                continue
            circles.append((int(xc), int(yc), int(r)))

        # ---- 視覺化 ----
        vis = frame.copy()
        for idx, (x, y, r) in enumerate(sorted(circles, key=lambda c: (c[1], c[0]))):
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 165, 255), -1)
            label = f"C{idx+1}"
            cv2.putText(vis, label, (x - 20, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(vis, f"found: {len(circles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ---- FPS ----
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(vis, f"FPS: {fps:.1f}", (W - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Circle Detector with Labels", vis)

        # 按 q 或 ESC 離開
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
