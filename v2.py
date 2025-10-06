import cv2, json, numpy as np, pandas as pd

IMG = "board.jpg"
CALIB_FILE="calib.yaml"
COLS, ROWS = 3, 4

# ---- 相機參數 (請換成實際內參) ----
fx, fy = 1200, 1200   # 焦距 (像素)
cx, cy = 640, 480     # 光心 (像素)
camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
dist_coeffs = np.zeros(5)  # 若有畸變請填入

# ---- 世界座標 (例：3×4 孔在平面上，單位 mm) ----
# 假設孔間距 50mm，左上為 (0,0)
objp = []
for j in range(ROWS):
    for i in range(COLS):
        objp.append([i*50.0, j*50.0, 0.0])
objp = np.array(objp, dtype=np.float32)

# ---- 偵測圓孔 (與前面版本相同，但包成函式) ----
HSV_TRIES = [((0,0,210),(179,40,255)), ((0,0,190),(179,60,255)), ((0,0,170),(179,80,255))]
KERNEL = (7,7)

def kmeans_1d(vals,k,iters=40):
    vals=np.asarray(vals,dtype=np.float32)
    c=np.linspace(float(vals.min()),float(vals.max()),k)
    for _ in range(iters):
        d=np.abs(vals[:,None]-c[None,:])
        lab=np.argmin(d,axis=1)
        for i in range(k):
            pts=vals[lab==i]
            if len(pts)>0: c[i]=np.median(pts)
    return sorted(c)

def circularity(cnt):
    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)
    if peri==0: return 0,area
    return 4*np.pi*area/(peri*peri), area

def detect_circles(img):
    H,W=img.shape[:2]; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel=np.ones(KERNEL,np.uint8)
    for low,high in HSV_TRIES:
        mask=cv2.inRange(hsv,low,high)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,2)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cands=[]
        for c in cnts:
            circ,area=circularity(c)
            if circ<0.78: continue
            (x,y),r=cv2.minEnclosingCircle(c)
            cands.append((int(x),int(y),int(r)))
        if len(cands)>0: return cands
    # 備援 Hough
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),1.2)
    c=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1.2,minDist=50,
                       param1=110,param2=24,minRadius=15,maxRadius=80)
    return np.round(c[0]).astype(int).tolist() if c is not None else []

def choose_grid(cands,H,W):
    xs=np.array([x for x,_,_ in cands],np.float32)
    ys=np.array([y for _,y,_ in cands],np.float32)
    colc=kmeans_1d(xs,COLS); rowc=kmeans_1d(ys,ROWS)
    grid={}
    for x,y,r in cands:
        ci=np.argmin([abs(x-cx) for cx in colc])
        rj=np.argmin([abs(y-cy) for cy in rowc])
        gcx,gcy=colc[ci],rowc[rj]; dist=np.hypot(x-gcx,y-gcy)
        if (ci,rj) not in grid or dist<grid[(ci,rj)][-1]:
            grid[(ci,rj)]=(x,y,r,dist)
    out=[]
    for rj in range(ROWS):
        for ci in range(COLS):
            x,y,r,_=grid[(ci,rj)]
            out.append((int(x),int(y),int(r)))
    return out

# ----------------- 主程式 -----------------
img=cv2.imread(IMG); H,W=img.shape[:2]
cands=detect_circles(img)
if len(cands)==0: raise SystemExit("未找到圓孔")

picked=choose_grid(cands,H,W)
imgp=np.array([(x,y) for x,y,_ in picked],dtype=np.float32)

# ---- PnP ----
ok,rvec,tvec=cv2.solvePnP(objp,imgp,camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
if not ok: raise SystemExit("PnP 失敗")

# ---- 繪製 ----
vis=img.copy()
for (x,y,r),(X,Y,Z) in zip(picked,objp):
    cv2.circle(vis,(x,y),r,(0,255,0),2)
    cv2.putText(vis,f"({X:.0f},{Y:.0f})",(x-20,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
cv2.imwrite("detected_pose.png",vis)

# ---- 輸出 ----
out={"rvec":rvec.flatten().tolist(),"tvec":tvec.flatten().tolist()}
with open("pose.json","w") as f: json.dump(out,f,indent=2)
pd.DataFrame([(x,y,r) for x,y,r in picked],columns=["x","y","r"]).to_csv("white_circles_xyr.csv",index=False)

print("saved: detected_pose.png, pose.json, white_circles_xyr.csv")
