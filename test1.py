import cv2, json, numpy as np, pandas as pd

IMG = "board.jpg"
COLS, ROWS = 3, 4

# ---- 參數：先嚴後鬆，逐步退讓 ----
HSV_TRIES = [
    ((0, 0, 210), (179, 40, 255)),  # 原本
    ((0, 0, 190), (179, 60, 255)),  # 放寬一點
    ((0, 0, 170), (179, 80, 255)),  # 再放寬
]
KERNEL = (7, 7)

def kmeans_1d(vals, k, iters=40):
    vals = np.asarray(vals, dtype=np.float32)
    c = np.linspace(float(vals.min()), float(vals.max()), k)
    for _ in range(iters):
        d = np.abs(vals[:, None] - c[None, :])
        lab = np.argmin(d, axis=1)
        moved = False
        for i in range(k):
            pts = vals[lab == i]
            if len(pts) > 0:
                newc = float(np.median(pts))
                if abs(newc - c[i]) > 1e-3:
                    c[i] = newc; moved = True
        if not moved: break
    return sorted(c)

def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return 0.0, area
    return float(4*np.pi*area/(peri*peri)), float(area)

def detect_by_hsv(img):
    """回傳候選 list[(x,y,r)] 與使用到的 (low,high)。"""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 從影像能量估一個保守的 beam y 上限（底部藍光很亮）
    v = hsv[:,:,2].astype(np.float32) / 255.0
    row_energy = v.mean(axis=1)
    # 取能量突升的 85 百分位當界線，避免過低
    thresh_row = int(np.clip(np.percentile(np.where(row_energy > row_energy.mean()*1.15)[0], 85, method="nearest"), 0, H-1))
    beam_y_max = max(int(H*0.78), min(H-1, thresh_row))

    kernel = np.ones(KERNEL, np.uint8)
    for low, high in HSV_TRIES:
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        min_area = 0.0006 * H * W
        max_area = 0.10   * H * W
        for c in cnts:
            circ, area = circularity(c)
            if area < min_area or area > max_area: 
                continue
            if circ < 0.78:
                continue
            (xc, yc), r = cv2.minEnclosingCircle(c)
            if yc > beam_y_max:  # 排除藍光區
                continue
            r = float(r)
            if r < 18 or r > min(H,W)*0.16:
                continue
            cands.append((int(round(xc)), int(round(yc)), int(round(r))))
        if len(cands) > 0:
            return cands, (low, high), beam_y_max
    return [], None, beam_y_max

def backup_hough(img, expect_n=12):
    """灰階 Hough 備援，之後用白度過濾。"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1.2)
    c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                         minDist=int(min(H,W)*0.08), param1=110, param2=24,
                         minRadius=int(min(H,W)*0.02), maxRadius=int(min(H,W)*0.20))
    if c is None:
        return []
    raw = np.round(c[0]).astype(int).tolist()

    # 以圓內像素的白度(HSV V)做過濾
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    kept = []
    for x,y,r in raw:
        x0,x1 = max(0, x-r), min(W-1, x+r)
        y0,y1 = max(0, y-r), min(H-1, y+r)
        roi = V[y0:y1, x0:x1].astype(np.float32)
        if roi.size == 0: continue
        # 圓形遮罩
        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        mask = (xx-(x-x0))**2 + (yy-(y-y0))**2 <= (r*0.85)**2
        mean_v = roi[mask].mean() if np.any(mask) else 0
        if mean_v > 200:   # 亮才保留
            kept.append((x,y,r))
    # 取前 expect_n 個（靠近 3×4 幾何再挑）
    return kept[:max(expect_n, 12)]

def choose_grid_3x4(cands, H, W):
    """把候選分配到 3×4 格，回傳 12 個 (x,y,r)。"""
    xs = np.array([x for x,_,_ in cands], np.float32)
    ys = np.array([y for _,y,_ in cands], np.float32)
    col_centers = kmeans_1d(xs, COLS)
    row_centers = kmeans_1d(ys, ROWS)

    grid = {}
    for x,y,r in cands:
        ci = int(np.argmin([abs(x - cx) for cx in col_centers]))
        rj = int(np.argmin([abs(y - cy) for cy in row_centers]))
        gcx, gcy = col_centers[ci], row_centers[rj]
        dist = np.hypot(x-gcx, y-gcy) / (r+1e-3)
        val = (x,y,r,dist)
        if (ci,rj) not in grid or dist < grid[(ci,rj)][-1]:
            grid[(ci,rj)] = val
    # 若有缺格，用最近的補上
    rest = [c for c in cands]
    for rj in range(ROWS):
        for ci in range(COLS):
            if (ci,rj) not in grid:
                gcx, gcy = col_centers[ci], row_centers[rj]
                best, bestd = None, 1e9
                for x,y,r in rest:
                    d = np.hypot(x-gcx, y-gcy) / (r+1e-3)
                    if d < bestd:
                        best, bestd = (x,y,r,d), d
                grid[(ci,rj)] = best
    # 排序輸出
    out = []
    for rj in range(ROWS):
        for ci in range(COLS):
            x,y,r,_ = grid[(ci,rj)]
            out.append((int(x),int(y),int(r)))
    return out

# ----------------- 主程式 -----------------
img = cv2.imread(IMG)
if img is None: raise SystemExit(f"讀不到 {IMG}")
H, W = img.shape[:2]

cands, used_range, beam_y_max = detect_by_hsv(img)
if len(cands) == 0:
    print("[WARN] HSV 法找不到候選，改用灰階 Hough 備援。")
    cands = backup_hough(img)

if len(cands) == 0:
    raise SystemExit("[ERROR] 仍找不到圓，請檢查影像或放寬門檻。")

# 依 3×4 幾何挑 12 個
picked = choose_grid_3x4(cands, H, W)

# 視覺化與輸出
vis = img.copy()
targets = []
for j in range(ROWS):
    for i in range(COLS):
        idx = j*COLS + i
        x,y,r = picked[idx]
        cv2.circle(vis, (x,y), r, (0,255,0), 2)
        cv2.circle(vis, (x,y), 2, (0,165,255), -1)
        tag = f"C{i}R{j}_{'small' if j==0 else 'big'}"
        cv2.putText(vis, tag, (x-24, y-r-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        targets.append({"id": tag, "x": x, "y": y, "r": r})

cv2.imwrite("detected_circles.png", vis)
print("saved: detected_circles.png")

pd.DataFrame([(t["x"], t["y"], t["r"]) for t in targets], columns=["x","y","r"]).to_csv("white_circles_xyr.csv", index=False)
with open("targets.json","w",encoding="utf-8") as f:
    json.dump({"targets": targets}, f, indent=2, ensure_ascii=False)
print("saved: white_circles_xyr.csv, targets.json")
