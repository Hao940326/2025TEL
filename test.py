import cv2
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

img = cv2.imread("board.jpg")
h, w = img.shape[:2]

# 1) 用 HSV 抓「白色」(低飽和+高亮度)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(hsv)

# 白色的條件：飽和度低、亮度高
mask_white = cv2.inRange(hsv, (0,0,215), (179,40,255))

# 2) 移除很小的雜點、填洞
kernel = np.ones((7,7), np.uint8)
mask_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

# 3) 找輪廓 + 依圓形度與面積篩選
cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0: 
        return 0, area
    circ = 4*np.pi*area/(peri*peri)
    return circ, area

candidates = []
for c in cnts:
    circ, area = circularity(c)
    if area < 1200 or area > 130000:  # 排除太小/太大
        continue
    if circ < 0.78:                    # 圓形度門檻
        continue
    # 以最小外接圓估半徑與中心
    (x,y), r = cv2.minEnclosingCircle(c)
    # 位置約束：排除下方藍色光束區(非白孔)
    if y > h*0.8:
        continue
    candidates.append((int(x), int(y), int(r)))

# 4) 顯示偵測結果
vis = img.copy()
for (x,y,r) in candidates:
    cv2.circle(vis, (x,y), r, (0,255,0), 3)
    cv2.circle(vis, (x,y), 2, (0,165,255), -1)

plt.figure(figsize=(9,11))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(9, 11))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.imwrite('detected_circles.png', vis)
