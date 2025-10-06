import cv2, numpy as np, glob, os, json

# === 參數 ===
IMG_DIR = "calib_images"            # 放棋盤照片的資料夾
PATTERN_SIZE = (9, 6)               # 內角點數 (cols, rows)
SQUARE_SIZE_MM = 25.0               # 單格邊長 (mm)
OUT_YAML = "calib.yaml"

# 建立 3D 物點 (Z=0 平面)
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints, imgpoints = [], []
h, w = None, None

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                glob.glob(os.path.join(IMG_DIR, "*.png")))

if not images:
    raise SystemExit(f"請把棋盤照片放到資料夾：{IMG_DIR}")

for fn in images:
    img = cv2.imread(fn)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if h is None:
        h, w = gray.shape[:2]
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        objpoints.append(objp)
        vis = cv2.drawChessboardCorners(img.copy(), PATTERN_SIZE, corners2, ret)
        cv2.imwrite(os.path.join(IMG_DIR, "_chk_"+os.path.basename(fn)), vis)

print(f"找到有效影像：{len(objpoints)} 張")
if len(objpoints) < 10:
    raise SystemExit("有效張數太少（<10），請多拍幾張不同角度的棋盤。")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print("RMS reprojection error:", ret)
print("K=\n", K)
print("dist=", dist.ravel())

# 存成 OpenCV YAML（供主程式讀取）
fs = cv2.FileStorage(OUT_YAML, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K)
fs.write("dist_coeffs", dist)
fs.release()
print("saved:", OUT_YAML)
