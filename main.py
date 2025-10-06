# main_geo_roi_fixed_gray_ransac_rowfinal_pnp.py
# 灰階→(寬鬆)Hough 估 3x4 幾何 → 兩端距離算 px/cm(含防呆/一致性檢查/離群濾除)
# → 依列別(0~3)設定 ROI/閾值/容忍度 → ROI 內 Canny+RANSAC 擬圓（Hough 備援）
# → 輸出 12 個圓(centers, r) 到 targets.json
# → 距離估計(視覺跨距+焦距像素) → distance.json
# → 若提供相機內參(標定) → solvePnP 估位姿 → distance_pose.json

import os, json, cv2, numpy as np

# ===== 檔案 =====
IMG        = "board.jpg"
CALIB_FILE = "calib.yaml"     # 可換成 "calib.json"；沒有就自動略過 PnP

# ===== 實際尺寸 (cm) =====
D_SMALL, D_BIG = 20.0, 40.0   # 上排小圓直徑20，其他大圓40
SX_SMALL, SY   = 70.0, 20.0   # 欄距70、列距20（欄：左中右；列：上到下）
COLS, ROWS     = 3, 4

# ===== Hough(第一次) 參數 =====
P2_LOOSE    = 22
MINR_RATIO  = 0.02
MAXR_RATIO  = 0.22

# ===== 列別化（由上到下 0~3 列）=====
ROI_SCALE_ROW   = [1.35, 1.35, 1.22, 1.10]
ADAPT_BLOCK_ROW = [41,   41,   55,   71]
ADAPT_C_ROW     = [7,    7,    9,    10]
EDGE_TOL_ROW    = [0.12, 0.12, 0.10, 0.08]
R_ERR_TOL_ROW   = [0.30, 0.30, 0.25, 0.22]
P2_REFINE_ROW   = [28,   28,   26,   24]
CENTER_LIM_FACT = 0.45

# ===== 通用 =====
CANNY_LO, CANNY_HI = 60, 160
R_TOL              = 0.12

# ===== 預覽輸出大小 =====
PREV_MAX_W, PREV_MAX_H = 1280, 720

# ===== 距離估計（視覺跨距用）焦距 =====
# 1) 直接填焦距(像素)；若不填，使用 2)
F_PIX = None
# 2) 以焦距(mm) + 感測器寬(mm) 自動換算
F_MM        = 4.2      # 沒有可留 None
SENSOR_W_MM = 6.4      # 沒有可留 None


# ---------- 小工具 ----------
def resize_preview(img, mw=1280, mh=720):
    h, w = img.shape[:2]
    s = min(mw / w, mh / h, 1.0)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img

def kmeans_1d(vals, k, iters=30):
    vals = np.asarray(vals, dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    centers = np.linspace(mn, mx, k)
    for _ in range(iters):
        d = np.abs(vals[:, None] - centers[None, :])
        lbl = np.argmin(d, axis=1)
        moved = False
        for i in range(k):
            pts = vals[lbl == i]
            if len(pts) > 0:
                c = float(np.median(pts))
                if abs(c - centers[i]) > 1e-3:
                    centers[i] = c; moved = True
        if not moved:
            break
    return sorted(centers)

def refine_in_roi(gray, cx, cy, r_expect, row, rng=None):
    """在 (cx,cy) 周邊小 ROI 內找最接近 r_expect 的圓心；RANSAC 擬圓，Hough 備援。"""
    H, W = gray.shape[:2]
    ROI_SCALE   = ROI_SCALE_ROW[row]
    ADAPT_BLOCK = ADAPT_BLOCK_ROW[row]
    ADAPT_C     = ADAPT_C_ROW[row]
    EDGE_TOL    = EDGE_TOL_ROW[row]
    R_ERR_TOL   = R_ERR_TOL_ROW[row]
    P2_REFINE   = P2_REFINE_ROW[row]
    bias = 0.20 if row <= 1 else 0.30

    roi_half = int(round(r_expect * ROI_SCALE))
    x0 = max(0, cx - roi_half); y0 = max(0, cy - roi_half)
    x1 = min(W - 1, cx + roi_half); y1 = min(H - 1, cy + roi_half)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return cx, cy

    base = cv2.bilateralFilter(roi, d=7, sigmaColor=60, sigmaSpace=7) if row >= 2 else roi
    rb   = cv2.GaussianBlur(base, (5, 5), 1.2)

    thr  = cv2.adaptiveThreshold(rb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C)
    edges = cv2.Canny(rb, CANNY_LO, CANNY_HI)
    edges = cv2.bitwise_and(edges, edges, mask=thr)

    ys, xs = np.where(edges > 0)
    if len(xs) > 30:
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        if rng is None: rng = np.random.default_rng(0)

        def fit_circle_3(p):
            (x1,y1),(x2,y2),(x3,y3) = p
            a = 2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
            if abs(a) < 1e-6: return None
            b = (x1**2+y1**2)*(y2-y3)+(x2**2+y2**2)*(y3-y1)+(x3**2+y3**2)*(y1-y2)
            c = (x1**2+y1**2)*(x3-x2)+(x2**2+y2**2)*(x1-x3)+(x3**2+y3**2)*(x2-x1)
            xc, yc = b/a, c/a
            r = np.hypot(pts[:,0]-xc, pts[:,1]-yc).mean()
            return xc, yc, r

        best = None; best_inl = -1
        iters = min(300, len(pts)*2)
        for _ in range(iters):
            idx = rng.choice(len(pts), 3, replace=False)
            res = fit_circle_3(pts[idx])
            if res is None: continue
            xc, yc, r = res
            if abs(r - r_expect) > r_expect * R_ERR_TOL: continue
            if np.hypot(xc - (cx - x0), yc - (cy - y0)) > r_expect * CENTER_LIM_FACT: continue
            d = np.abs(np.hypot(pts[:,0]-xc, pts[:,1]-yc) - r)
            inl = int((d < r_expect * EDGE_TOL).sum())
            if inl > best_inl:
                best_inl, best = inl, (xc, yc)

        if best is not None:
            gx = int(round(best[0])) + x0
            gy = int(round(best[1])) + y0
            gx = int(round(gx * (1 - bias) + cx * bias))
            gy = int(round(gy * (1 - bias) + cy * bias))
            return gx, gy

    # 窄帶 Hough 備援
    rmin = int(max(3, r_expect*(1-R_TOL)))
    rmax = int(max(rmin+2, r_expect*(1+R_TOL)))
    c = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2,
                         minDist=int(r_expect*0.9), param1=120, param2=P2_REFINE,
                         minRadius=rmin, maxRadius=rmax)
    if c is not None:
        bx, by, bd = None, None, 1e9
        for x,y,_ in np.round(c[0]).astype(int):
            gx, gy = x + x0, y + y0
            d2 = (gx - cx)**2 + (gy - cy)**2
            if d2 < bd and np.sqrt(d2) <= r_expect * CENTER_LIM_FACT:
                bd, bx, by = d2, gx, gy
        if bx is not None:
            bx = int(round(bx * (1 - bias) + cx * bias))
            by = int(round(by * (1 - bias) + cy * bias))
            return int(bx), int(by)

    return cx, cy


# ----------------- 主流程 -----------------
# 1) 讀圖→灰階→模糊
img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"讀不到 {IMG}")
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5,5), 1.2)

# 2) 初次（寬鬆）Hough：只用來估欄/列中心
minR, maxR = int(min(H,W)*MINR_RATIO), int(min(H,W)*MAXR_RATIO)
c = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2,
                     minDist=int(min(H,W)*0.06), param1=110, param2=P2_LOOSE,
                     minRadius=minR, maxRadius=maxR)
if c is None or len(c[0]) < 8:
    cv2.imwrite("debug_blur.png", gray_blur)
    raise SystemExit(f"候選太少({0 if c is None else len(c[0])})；可降 P2_LOOSE 或提高解析度。")
cands = [(int(x),int(y),int(r)) for x,y,r in np.round(c[0]).astype(int)]

# 2.1 半徑離群濾除
rs = np.array([r for _,_,r in cands], dtype=np.float32)
med = float(np.median(rs))
iqr = float(np.percentile(rs, 75) - np.percentile(rs, 25))
lo = med - 2.0*iqr; hi = med + 2.0*iqr
cands = [(x,y,r) for (x,y,r) in cands if lo <= r <= hi]

xs = np.array([x for x,_,_ in cands], dtype=np.float32)
ys = np.array([y for _,y,_ in cands], dtype=np.float32)
cx = kmeans_1d(xs, COLS)  # 左中右
cy = kmeans_1d(ys, ROWS)  # 上到下

# 3) 兩端距離算 px/cm（含合理範圍檢查 + 一致性檢查 + 回退）
px_per_cm_h = abs(cx[2] - cx[0]) / 140.0     # 左↔右 ≈ 140 cm
px_per_cm_v = abs(cy[3] - cy[0]) /  60.0     # 上↔下 ≈  60 cm
low_h = 0.45 * W / 140.0;  hi_h = 0.90 * W / 140.0
low_v = 0.35 * H /  60.0;  hi_v = 0.90 * H /  60.0
def clamp(v, lo, hi): return max(lo, min(hi, v))
pxh = clamp(px_per_cm_h, low_h, hi_h)
pxv = clamp(px_per_cm_v, low_v, hi_v)
ratio = max(pxh, pxv) / max(1e-6, min(pxh, pxv))
px_per_cm = min(pxh, pxv) if ratio > 1.8 else float(np.median([pxh, pxv]))

R_SMALL = (D_SMALL/2.0) * px_per_cm
R_BIG   = (D_BIG/2.0)   * px_per_cm
min_r   = 0.03 * min(H, W)
max_r_s = 0.12 * min(H, W)
max_r_b = 0.24 * min(H, W)
if not (min_r <= R_SMALL <= max_r_s and min_r*2 <= R_BIG <= max_r_b):
    px_per_cm = pxh
    R_SMALL = (D_SMALL/2.0) * px_per_cm
    R_BIG   = (D_BIG/2.0)   * px_per_cm

print(f"px/cm_h={px_per_cm_h:.3f}(→{pxh:.3f})  px/cm_v={px_per_cm_v:.3f}(→{pxv:.3f})  ratio={ratio:.2f}  → 使用 {px_per_cm:.3f}")
print(f"期望半徑 small={R_SMALL:.1f}px  big={R_BIG:.1f}px")
print(f'cx: [{", ".join(f"{v:.1f}" for v in cx)}]  span_x: {cx[2]-cx[0]:.1f} px')
print(f'cy: [{", ".join(f"{v:.1f}" for v in cy)}]  span_y: {cy[3]-cy[0]:.1f} px')

# 4) 3×4 網格交點
grid = [(i, j, int(round(cx[i])), int(round(cy[j])))
        for j in range(ROWS) for i in range(COLS)]

# 5) 各格 ROI 精修（第 0 列 small，其餘 big；傳入 row=j）
rng = np.random.default_rng(0)
targets = []
for (i,j,x,y) in grid:
    r_exp = (D_SMALL/2.0 if j==0 else D_BIG/2.0) * px_per_cm
    rx, ry = refine_in_roi(gray, x, y, r_exp, j, rng)
    targets.append({"id": f"C{i}R{j}_{'small' if j==0 else 'big'}",
                    "x": int(rx), "y": int(ry), "r": int(round(r_exp))})

# 6) 輸出與視覺化
os.makedirs("out", exist_ok=True)
with open(os.path.join("out","targets.json"),"w", encoding="utf-8") as f:
    json.dump({"targets": targets}, f, indent=2, ensure_ascii=False)
print("saved out/targets.json:", len(targets))

vis = img.copy()
for t in targets:
    color = (0,255,255) if "small" in t["id"] else (0,255,0)
    cv2.circle(vis, (t["x"], t["y"]), t["r"], color, 2)
    cv2.putText(vis, t["id"], (t["x"]-20, t["y"]-t["r"]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

prev = resize_preview(vis, PREV_MAX_W, PREV_MAX_H)
cv2.imwrite(os.path.join("out","check_targets_full.png"), vis)
cv2.imwrite(os.path.join("out","preview.png"), prev)

# ===== 7) 距離估計（視覺跨距 + 焦距像素）==============================
if F_PIX is None and (F_MM and SENSOR_W_MM):
    F_PIX = (F_MM / SENSOR_W_MM) * W

if F_PIX is None:
    print("[WARN] 未提供 F_PIX/F_MM/SENSOR_W_MM，略過絕對距離(visual)。")
else:
    span_x_px = abs(cx[2] - cx[0])     # 對應實寬 140 cm
    span_y_px = abs(cy[3] - cy[0])     # 對應實高  60 cm
    Zx = (F_PIX * 140.0) / max(1e-6, span_x_px)
    Zy = (F_PIX *  60.0) / max(1e-6, span_y_px)
    ratio_Z = max(Zx, Zy) / max(1e-6, min(Zx, Zy))
    Z = np.median([Zx, Zy]) if ratio_Z < 1.6 else (Zx if span_x_px/W > span_y_px/H else Zy)

    u_c = float(np.mean([t["x"] for t in targets]))
    v_c = float(np.mean([t["y"] for t in targets]))
    cx0, cy0 = W/2.0, H/2.0
    X = (u_c - cx0) * Z / F_PIX
    Y = (v_c - cy0) * Z / F_PIX

    dist_info = {
        "f_pix": float(F_PIX),
        "span_x_px": float(span_x_px),
        "span_y_px": float(span_y_px),
        "Z_from_width_cm": float(Zx),
        "Z_from_height_cm": float(Zy),
        "Z_cm": float(Z),
        "board_center_px": [float(u_c), float(v_c)],
        "principal_point_px": [float(cx0), float(cy0)],
        "X_cm": float(X),
        "Y_cm": float(Y)
    }
    with open(os.path.join("out","distance.json"), "w", encoding="utf-8") as f:
        json.dump(dist_info, f, indent=2, ensure_ascii=False)
    print(f"[Distance] Z={Z:.2f} cm, X={X:.2f} cm, Y={Y:.2f} cm (f_pix={F_PIX:.1f})")
    print("saved out/distance.json")

# ===== 8) 若有相機內參 → PnP 求位姿 =====================================
def _load_intrinsics(path):
    if not os.path.exists(path): return None, None
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            j = json.load(f)
        K = np.array(j["camera_matrix"], dtype=np.float64)
        dist = np.array(j.get("dist_coeffs", []), dtype=np.float64).reshape(-1,1) if "dist_coeffs" in j else None
        return K, dist
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened(): return None, None
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("dist_coeffs").mat()
        fs.release()
        if dist is not None: dist = dist.reshape(-1,1)
        return K, dist

K, dist = _load_intrinsics(CALIB_FILE)
if K is None:
    print(f"[INFO] 找不到或讀不到內參檔：{CALIB_FILE}，略過 PnP。")
else:
    obj_pts, img_pts = [], []
    COL_SP, ROW_SP = 70.0, 20.0
    for t in targets:
        sid = t["id"]
        i = int(sid[sid.find("C")+1 : sid.find("R")])
        j = int(sid[sid.find("R")+1 : sid.find("_")])
        Xw = (i - 1) * COL_SP     # 中欄 = 0
        Yw = j * ROW_SP           # 上到下 0,20,40,60
        obj_pts.append([Xw, Yw, 0.0])
        img_pts.append([t["x"], t["y"]])
    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts, imagePoints=img_pts,
        cameraMatrix=K, distCoeffs=dist,
        reprojectionError=2.0, flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=300
    )
    if not ok:
        print("[ERROR] solvePnPRansac 失敗。")
    else:
        ok2, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts[inliers[:,0]],
            imagePoints=img_pts[inliers[:,0]],
            cameraMatrix=K, distCoeffs=dist,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        R,_ = cv2.Rodrigues(rvec)
        yaw   = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
        pitch = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))))
        roll  = float(np.degrees(np.arctan2(R[2,1], R[2,2])))

        pose = {
            "inliers": int(len(inliers)),
            "tvec_cm": [float(tvec[0]), float(tvec[1]), float(tvec[2])],  # X,Y,Z (cm)
            "euler_deg": {"yaw": yaw, "pitch": pitch, "roll": roll}
        }
        with open(os.path.join("out","distance_pose.json"), "w", encoding="utf-8") as f:
            json.dump(pose, f, indent=2, ensure_ascii=False)
        print(f"[PnP] Z={tvec[2][0]:.2f} cm, X={tvec[0][0]:.2f} cm, Y={tvec[1][0]:.2f} cm | "
              f"yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}° (inliers={len(inliers)})")
        print("saved out/distance_pose.json")
