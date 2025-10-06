import cv2, json, numpy as np, pandas as pd, os

IMG = "board.jpg"
CALIB_FILE = "calib.yaml"      # 你的 C270 標定檔
COLS, ROWS = 3, 4

# ---- 世界座標 (例：3×4 孔在平面上，單位 mm) ----
# TODO: 依實際板子量測修改！現在先用等距 50 mm 範例。
objp = []
GRID_MM_X = 50.0
GRID_MM_Y = 50.0
for j in range(ROWS):
    for i in range(COLS):
        objp.append([i*GRID_MM_X, j*GRID_MM_Y, 0.0])
objp = np.array(objp, dtype=np.float32)

# ---- 偵測圓孔（HSV 多組門檻 + Hough 備援）----
HSV_TRIES = [((0,0,210),(179,40,255)), ((0,0,190),(179,60,255)), ((0,0,170),(179,80,255))]
KERNEL = (7,7)

def kmeans_1d(vals,k,iters=40):
    vals=np.asarray(vals,dtype=np.float32)
    c=np.linspace(float(vals.min()),float(vals.max()),k)
    for _ in range(iters):
        d=np.abs(vals[:,None]-c[None,:])
        lab=np.argmin(d,axis=1)
        moved=False
        for i in range(k):
            pts=vals[lab==i]
            if len(pts)>0:
                newc=float(np.median(pts))
                if abs(newc-c[i])>1e-3:
                    c[i]=newc; moved=True
        if not moved: break
    return sorted(c)

def circularity(cnt):
    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)
    if peri==0: return 0.0, area
    return float(4*np.pi*area/(peri*peri)), float(area)

def detect_circles(img):
    """回傳 [(x,y,r)]；先 HSV，失敗則 Hough 備援"""
    H,W=img.shape[:2]; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel=np.ones(KERNEL,np.uint8)
    for low,high in HSV_TRIES:
        mask=cv2.inRange(hsv,low,high)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,2)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cands=[]
        min_area=0.0006*H*W; max_area=0.10*H*W
        for c in cnts:
            circ,area=circularity(c)
            if area<min_area or area>max_area: continue
            if circ<0.78: continue
            (x,y),r=cv2.minEnclosingCircle(c)
            r=float(r)
            if r<15 or r>min(H,W)*0.18: continue
            cands.append((int(round(x)),int(round(y)),int(round(r))))
        if len(cands)>0:
            return cands
    # Hough 備援
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),1.2)
    c=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=int(min(H,W)*0.08),
                       param1=110,param2=24,minRadius=int(min(H,W)*0.02),maxRadius=int(min(H,W)*0.20))
    return np.round(c[0]).astype(int).tolist() if c is not None else []

def choose_grid(cands,H,W):
    """把候選分配到 3×4 格；若缺格則用最近的補上。回傳 12 個 (x,y,r)"""
    if len(cands)==0:
        raise RuntimeError("沒有候選圓可分配")
    xs=np.array([x for x,_,_ in cands],np.float32)
    ys=np.array([y for _,y,_ in cands],np.float32)
    if len(xs)<COLS or len(ys)<ROWS:
        # 兜底：用畫面均勻切分估中心，避免 kmeans 崩
        colc=[W*(i+0.5)/COLS for i in range(COLS)]
        rowc=[H*(j+0.5)/ROWS for j in range(ROWS)]
    else:
        colc=kmeans_1d(xs,COLS); rowc=kmeans_1d(ys,ROWS)

    grid={}
    for x,y,r in cands:
        ci=int(np.argmin([abs(x-cx) for cx in colc]))
        rj=int(np.argmin([abs(y-cy) for cy in rowc]))
        gcx,gcy=colc[ci],rowc[rj]
        dist=np.hypot(x-gcx,y-gcy)/(r+1e-3)
        if (ci,rj) not in grid or dist<grid[(ci,rj)][-1]:
            grid[(ci,rj)]=(x,y,r,dist)

    # 補缺格
    if len(grid)<COLS*ROWS:
        pool=list(cands)
        for rj in range(ROWS):
            for ci in range(COLS):
                if (ci,rj) in grid: continue
                gcx,gcy=colc[ci],rowc[rj]
                best=None; bestd=1e9
                for x,y,r in pool:
                    d=np.hypot(x-gcx,y-gcy)/(r+1e-3)
                    if d<bestd:
                        best=(x,y,r,d); bestd=d
                grid[(ci,rj)]=best

    out=[]
    for rj in range(ROWS):
        for ci in range(COLS):
            x,y,r,_=grid[(ci,rj)]
            out.append((int(x),int(y),int(r)))
    return out

def load_intrinsics(path):
    """讀 OpenCV YAML/JSON：回傳 K(3x3), dist(n,1)"""
    if not os.path.exists(path): return None, None
    if path.lower().endswith(".json"):
        with open(path,"r") as f:
            j=json.load(f)
        K=np.array(j["camera_matrix"],dtype=np.float64)
        dist=np.array(j.get("dist_coeffs",[]),dtype=np.float64).reshape(-1,1) if "dist_coeffs" in j else None
        return K, dist
    fs=cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): return None, None
    K=fs.getNode("camera_matrix").mat()
    dist=fs.getNode("dist_coeffs").mat()
    fs.release()
    if dist is not None: dist=dist.reshape(-1,1)
    return K, dist

def euler_zyx_from_R(R):
    """回傳 yaw(Z), pitch(Y), roll(X) in degrees"""
    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
    pitch = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)))
    roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
    return float(yaw), float(pitch), float(roll)

# ----------------- 主程式 -----------------
img=cv2.imread(IMG)
if img is None: raise SystemExit(f"讀不到 {IMG}")
H,W=img.shape[:2]

# 讀取相機內參，先去畸變
K, dist = load_intrinsics(CALIB_FILE)
if K is None:
    raise SystemExit(f"讀不到內參檔：{CALIB_FILE}")
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W,H), alpha=0.0, newImgSize=(W,H))
und = cv2.undistort(img, K, dist, None, newK)

# 偵測圓孔（用去畸變後的影像）
cands = detect_circles(und)
if len(cands)==0:
    raise SystemExit("未找到圓孔；請放寬門檻或檢查影像。")

picked = choose_grid(cands, H, W)
imgp = np.array([(x,y) for x,y,_ in picked], dtype=np.float32)

# ---- PnP：RANSAC + refine（使用 newK，畸變已去除→ dist=None）----
ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    objectPoints=objp,
    imagePoints=imgp,
    cameraMatrix=newK,
    distCoeffs=None,
    flags=cv2.SOLVEPNP_EPNP,
    reprojectionError=2.0,
    iterationsCount=300
)
if not ok:
    raise SystemExit("PnP RANSAC 失敗")

ok2, rvec, tvec = cv2.solvePnP(
    objectPoints=objp[inliers[:,0]],
    imagePoints=imgp[inliers[:,0]],
    cameraMatrix=newK, distCoeffs=None,
    rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# ---- 視覺化 ----
vis = und.copy()
for (x,y,r),(X,Y,Z) in zip(picked,objp):
    cv2.circle(vis,(x,y),r,(0,255,0),2)
    cv2.putText(vis,f"({X:.0f},{Y:.0f})",(x-24,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
cv2.imwrite("detected_pose.png", vis)

# ---- 轉成歐拉角 + 友善輸出 ----
R,_ = cv2.Rodrigues(rvec)
yaw,pitch,roll = euler_zyx_from_R(R)
tvec_mm = tvec.reshape(-1)               # mm
out = {
    "inliers": int(len(inliers)),
    "rvec": rvec.reshape(-1).tolist(),
    "tvec_mm": tvec_mm.tolist(),
    "tvec_cm": (tvec_mm/10.0).tolist(),
    "euler_deg": {"yaw": yaw, "pitch": pitch, "roll": roll}
}
with open("pose.json","w",encoding="utf-8") as f:
    json.dump(out,f,indent=2,ensure_ascii=False)

pd.DataFrame([(x,y,r) for x,y,r in picked],columns=["x","y","r"]).to_csv("white_circles_xyr.csv",index=False)

print(f"saved: detected_pose.png, pose.json, white_circles_xyr.csv")
print(f"[PnP] Z = {tvec_mm[2]:.1f} mm ({tvec_mm[2]/10.0:.2f} cm), "
      f"X = {tvec_mm[0]:.1f} mm, Y = {tvec_mm[1]:.1f} mm | "
      f"yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}° | inliers={len(inliers)}")