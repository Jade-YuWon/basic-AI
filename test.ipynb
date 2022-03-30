# 문제 3번 둥근 정도 출력

import numpy as np
import math
import cv2

from google.colab import files
uploaded = files.upload()

"""
A1_P3_detectCircle.png
A1_P3_detectCircle.png(image/png) - 289220 bytes, last modified: 2022. 3. 16. - 100% done
Saving A1_P3_detectCircle.png to A1_P3_detectCircle.png
"""

import os
from google.colab.patches import cv2_imshow
imgpath = os.path.join('./', 'A1_P3_detectCircle.png')
img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# circularity is 4pi(Area)/(Perimeter^2), while roundness is 4A/(pi*Major Axis^2)
output = img.copy()
thre = 71
blurred_gray = cv2.blur(gray, (5,5))
ret, binary_image = cv2.threshold(blurred_gray, thre, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

output = cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
for i, contour in enumerate(contours):
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, True)
  circularity = 4 * math.pi * area / math.pow(perimeter, 2) # 진원도
  print("[%d] %f"%(i, circularity))
  # 컨투어 첫 좌표에 인덱스 숫자 표시
  cv2.putText(output, str(circularity), tuple(contour[0][0]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
cv2_imshow(output)
