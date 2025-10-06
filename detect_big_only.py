import cv2, json, numpy as np
from math import hypot

IMG = "board.jpg"
COLS, ROWS = 3, 4             # 3欄4列
SY_CM = 20.0                  # 列距 20cm
D_BIG = 40.0                  # 大圓直徑 40cm

# ---- 可調參 ----
P2_LOOSE = 22                 # 第一次寬鬆 Hough
MINR_RATIO, MAXR_RATIO = 0.02, 0.22
P2_BIG = 28                   # 第二次（只抓大圓）的 Hough 門檻（越大越嚴）
R_TOL = 0.15                  # 大圓半徑窄帶 ±15%
PREV_MAX_W, PREV_MAX_H = 1280, 720

def resize_preview(img, mw=1280, mh=720):
    h, w = img.shape[:2]
    s = min(mw/w, mh/h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s))) if s < 1 else img

def kmeans_1d(vals, k, iters=30):
    vals = np.asarray(vals, dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    centers = np.linspace(mn, mx, k)
    for _ in range(iters):
        d = np.abs(vals[:, None]-centers[None, :])
        lbl = np.argmin(d, axis=1)
        changed = False
        for i in range(k):
            pts = vals[lbl==i]
            if len(pts) > 0:
                c = float(np.median(pts))
                if abs(c-centers[i]) > 1e-3:
                    centers[i] = c; changed = True
        if not changed: break
    return sorted(centers)

def hough_band(img_gray, r_center, tol_ratio, minDist_px, p2):
    rmin = max(3, int(r_center*(1-tol_ratio)))
    rmax = max(rmin+2, int(r_center*(1+tol_ratio)))
    c = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1.2,
                         minDist=minDist_px, param1=120, param2=p2,
                         minRadius=rmin, maxRadius=rmax)
    out=[]
    if c is not None:
        for x,y,r in np.round(c[0]).astype(int):
            out.append((int(x),int(y),int(r)))
    return out

# ---------- 讀圖 + 前處理 ----------
img = cv2.imread(IMG)
if img is None: raise SystemExit(f"讀不到 {IMG}")
H,W = img.shape[:2]
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g = cv2.createCLAHE(2.0,(8,8)).apply(g)
g = cv2.GaussianBlur(g,(5,5),1.2)

# ---------- 寬鬆 Hough（只為了找列心與比例） ----------
minR,maxR = int(min(H,W)*MINR_RATIO), int(min(H,W)*MAXR_RATIO)
c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                     minDist=int(min(H,W)*0.06), param1=110, param2=P2_LOOSE,
                     minRadius=minR, maxRadius=maxR)
if c is None or len(c[0]) < 8:
    cv2.imwrite("debug_blur.png", g)
    raise SystemExit("候選太少，調低 P2_LOOSE 或提高亮度/解析度")

cands = [(int(x),int(y),int(r)) for x,y,r in np.round(c[0]).astype(int)]
ys = np.array([y for _,y,_ in cands], dtype=np.float32)

# 分出 4 列中心（由上到下）
row_centers = kmeans_1d(ys, ROWS)         # [cy0, cy1, cy2, cy3]
row_centers = sorted(row_centers)

# 用相鄰列距估 px/cm（中位數）
d_v = [abs(row_centers[j+1]-row_centers[j]) for j in range(ROWS-1)]
px_per_cm = float(np.median(np.array(d_v, dtype=np.float32)) / SY_CM)
r_big_px = (D_BIG/2.0) * px_per_cm

# 只在列 1..3 區域搜尋大圓
mask = np.zeros((H,W), np.uint8)
top = int((row_centers[0] + row_centers[1]) / 2)     # 小列與第2列之間
bot = int(min(H-1, row_centers[-1] + (row_centers[1]-row_centers[0])//2))
cv2.rectangle(mask, (0, top), (W, bot), 255, -1)
g_big = cv2.bitwise_and(g, g, mask=mask)

# 第二次 Hough：窄半徑帶，只抓大圓
big_cands = hough_band(g_big, r_big_px, R_TOL,
                       minDist_px=int(px_per_cm*SY_CM*0.6),
                       p2=P2_BIG)

# 以 X 分 3 欄，並在每欄由上而下取 3 顆（大圓 9 顆）
xs = np.array([x for x,_,_ in big_cands], dtype=np.float32)
col_centers = kmeans_1d(xs, COLS)  # [cx0,cx1,cx2] 左中右
col_centers = sorted(col_centers)

def assign_col(x):
    diffs = [abs(x-cx) for cx in col_centers]
    return int(np.argmin(diffs))

cols = {0:[],1:[],2:[]}
for x,y,r in big_cands:
    cols[assign_col(x)].append((x,y,r))
for k in cols: cols[k] = sorted(cols[k], key=lambda t:t[1])[:3]  # 每欄取最上面的3顆

# 打包 9 個大孔（C0..2, R1..3）
targets=[]
for ci in range(COLS):
    for ri in range(1, ROWS):  # 1..3
        if ri-1 < len(cols[ci]):
            x,y,_ = cols[ci][ri-1]
        else:
            # 缺顆時，用交點近似
            x = int(col_centers[ci]); y = int(row_centers[ri])
        targets.append({"id": f"C{ci}R{ri}_big", "x": x, "y": y, "r": int(round(r_big_px))})

with open("targets_big.json","w",encoding="utf-8") as f:
    json.dump({"targets":targets}, f, indent=2)
print("saved targets_big.json:", len(targets), "px/cm≈", round(px_per_cm,3))

# 視覺化
vis = img.copy()
for t in targets:
    cv2.circle(vis,(t["x"],t["y"]),t["r"],(0,255,0),2)
    cv2.putText(vis,t["id"],(t["x"]-20,t["y"]-t["r"]-6),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
prev = resize_preview(vis, PREV_MAX_W, PREV_MAX_H)
cv2.imshow("big only", prev)
cv2.imwrite("check_big_full.png", vis)
cv2.imwrite("check_big_preview.png", prev)
cv2.waitKey(0); cv2.destroyAllWindows()
