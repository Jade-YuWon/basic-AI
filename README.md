# basic-CV
learning records of basic Computer Vision, such as python data analasis and pandas, etc.

# Long-term Study Plan
# 2022 ~
## - URA (Undergraduate Reasearch Assistant) of CVIP Lab
#### CVIP (Computer Vision & Image Processing) Lab of Prof. Jung Yong-Ju at Gachon University 
- https://sites.google.com/site/gachoncvip/home?authuser=0
- https://github.com/CVIP-LAB
#### Study group with fellow URA Kim
- Attend at the Lab Seminar once in 2 \~ 4 weeks
- __Learning the Basic of Computer Vision__
- Data Analasis: (inflearn)
  -  **Python**, Pandas, EDA, Jupiter notebook
  -  Matplotlib, Plotly Library (Visualization)
- Machine Learning: (inflearn)
  - Kaggle, Feature Engineering
  - Understand of Classification models, Performance Evaluation, Tuning
  - Regression / Clustering models
- Mathematics:
  - Linear Algebra (Khan Academy and Prof. Choo Jaegul)
  - Statitics and Probability
    - Matrix Cookbook https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
- Deep Learning:
  - CNN
  - **Pytorch**
  - *Vision Transformer* (*Expected)
## - Major subjects of Artificial Intelligence
  -  Data Structure and Practice with C
  -  OOP(Objective Oreiented Programming) with JAVA
  -  Operating Systems
  -  Probabilities and Statistics


# Short-term Study Contents
# 2021.12 ~ (Winter Vacation)
⭕ : Learned as planned out <br>
🔺 : Missed some part

## 21.12.28(Tue)
Start Study group activity ⭕<br>
Every Wed, Sat 10 PM
## ~22.01.01(Sat)
1. Python Data analasis (Part 1): ~Section 03 ⭕
2. Basic of Machine Learning (Part 2): ~Section 03 ⭕
3. Linear Algebra for Everyone: ~Ch. 01 ⭕
## ~22.01.05(Wed)~ ~01.12(Wed)
1. Python Data analasis (Part 1): ~Section 04
6. Basic of Machine Learning (Part 2): ~Section 04 (+ 05)
7. Linear Algebra for Everyone: ~Ch. 03 (+ 04)
7. Linear Algebra for AI: ~Ch. 02
## 22.01.30(Sun)
- Informal Graduate-Student applicant Seminar
  - The State of the Art(sota) - AI https://paperswithcode.com/sota
## 22.02.03(Thu)
1. 
2. 

## 22.02.28 (Mon)
1. Dilation and Stride
  - input image
  - sampling? grid
  - dilation rate
  - stride
2. magic filter
  - bilinear filter 이중선형 필터링
  - bilateral filter 양방향 필터

## 22.03.01 (Tue)
1. Non-linearity of CNN
  - Perceptron
  - multi layers, input/hidden/output
  - w(weights) tuning
  - linearity
  - bias
  - ground truth
  - activation function - **Sigmoid**
2. Event camera vs. normal camera
  - how shutter works
  - filter and color values
  - light and energy of photon
  - condition of event camera
  - price: expensive(event) / cheap(normal)
3. Medical segmentation in 2D, 3D, 4D...
  - Color image vs. Grayscale image
  - different bits of level(2^8 vs. 2^16)
  - voxel
  - spine overlayed analysis
4. etc
  - attitude: ~~cry then you success~~ always learn from others, discuss, throw yourself into what you're fascinated
  - If you're not coding in my dream, I would fail.
  - 학사 vs. 석사 vs. 박사
    - Bachelor: Now I know everything!
    - Master: What do I know?
    - Ph.D: I know nothing

## ?
- Event camera

## 22.03.06 (Sun)
1. How computer vision identify a circle?
- Geometric basic
  - Area of circle: pi r^2
  - Surface area of a Sphere: 4pi r^2
  - Volume of a Sphere: 4/3 pi r^3
- Popular apporoach: Hough Transformations
  - 허프 변환 https://en.wikipedia.org/wiki/Hough_transform
    - OpenCV 파이썬으로 만드는 OpenCV 프로젝트(이세우 저) https://bkshin.tistory.com/entry/OpenCV-23-%ED%97%88%ED%94%84-%EB%B3%80%ED%99%98Hough-Transformation
  - generalized Hough transform (GHT) https://en.wikipedia.org/wiki/Generalised_Hough_transform
  
- Paper
  - Circle detection on images by line segment and circle completeness: https://ieeexplore.ieee.org/document/7533040


## 22.03.12 (Sat)
- Computer vision vs. Computer Graphics
  - CV: vector input, image/video output
  - CG: image/video input, vector output

## 22.03.14 (Mon)
- Machine "not" learning
  - Machines don't learng anything. Engineers don't teach or train machine.
  - They just "approximate" and "estimate" parameters.

## 22.03.16 (Wed)
- Toy project such as 10 billion-unit calculator, ...
- Bio & medical CV: 성분 normalization, robot 다관절 제어, 반려견 목둘레 자동 측정, ...


## 22.03.18 (Fri)
- 대학진로탐색학점제 커리어캐치 AI프로젝트 - Pytorch Tutorial 복습
- 머신러닝 과정(Machine Learning process)
  1. Data processing
  2. Create Model 
  3. Model Optimazation
- ROI(Region Of Interest)
  - ROI는 뜻 그대로 **이미지나 영상 내에서 내가 관심있는 부분**을 뜻한다. 이미지 상의 특정 오브젝트나 특이점을 찾는 것을 목표로 할 때 쓴다. (비슷한 용어로는 COI(Channel Of Interest)라는 관심 채널이 있다.) 2020. 11. 24.
  - [OpenCV 튜토리얼 5. 관심영역 ROI(Region Of Interest)](https://eusun0830.tistory.com/42) 


## 22.03.19 (Sat)
1. OpenCV-Python Tutorial
> That said, Python can be easily extended with C/C++, which allows us to write computationally intensive code in C/C++ and create Python wrappers that can be used as Python modules.
  - Python wrappers
  - Python Module

2. Python image input
  - cv2.imread(fileName, flag)
    - gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
  - cv2.imshow(title, image)
      - title(str) – 윈도우 창의 Title  
      - image (numpy.ndarray) – cv2.imread() 의 return값
  - cv2.imwrite(fileName, image)
```python
from google.colab import files
uploaded = files.upload()
```

>  A1_P3_detectCircle.png<br>
> A1_P3_detectCircle.png(image/png) - 289220 bytes, last modified: 2022. 3. 16. - 100% done<br>
> Saving A1_P3_detectCircle.png to A1_P3_detectCircle.png
```python
import cv2
from google.colab.patches import cv2_imshow
img = cv2.imread(os.path.join('./', 'A1_P3_detectCircle.png'))
cv2_imshow(img)

```

3. Detect Circle
- [Official OpenCV Reference - Hough Circle Transform](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html)
- [Image Processing - Feature Detection - HoughCircles()](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d)
```python
# cv2.HoughCircles(image, method, dp, minDist)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
```
- image: 8-bit, single channel image. If working with a color image, convert to grayscale first.
  - single channel image
- method: Defines the method to detect circles in images. Currently, the only implemented method is cv2.HOUGH_GRADIENT, which corresponds to the [Yuen et al.](http://www.bmva.org/bmvc/1989/avc-89-029.pdf) paper.

4. Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")


## 22.03.22 (TUE)
1. Contour = 동일한 색 또는 동일한 픽셀값(강도, intensity)을 가지고 있는 영역의 경계선 정보. 물체 윤곽/외형 파악에 사용.
- contours, hierarchy = cv.findContours(binary_image, mode, method)
  - mode = cv2.RETR_LIST, cv2.RETR_EXTERNAL, ...
  - method = cv2.CHAIN_APPROX_NONE or cv2.CHAIN_APPROX_SIMPLE
  - Binary image (Image segmentation 이진화)
    - threshold(image_gray, thre, 255, cv2.THRESH_BINARY)
  - Moment
    - c0 = contours\[0\], M = cv2.moments(c0)
    - 중심점(Contour center point): cx = int(M\['m10'\]/M\['m00'\]),  cy = int(M\['m01'\]/M\['m00'\])
    - 면적(Contour area): M['m00'] 또는 cv2.contourArea(c0)
  - drawContours(img_color, contours, contourIdx, color\[, thickness])
    - contourIdx = -1: draw every contours to image
    - thickness = -1: fill the contour(s)
  - 참고글 https://bkshin.tistory.com/entry/OpenCV-22-%EC%BB%A8%ED%88%AC%EC%96%B4Contour
2. Blur
- blurred_gray = cv2.blur(gray, (5,5))
  - blurred_gray = cv2.GaussianBlur(gray, (5,5), 0)
  - Median blurring
  - bilateral filtering
  - Erosion, Dilation / opening, closing (Morphology)
3. cv, cv2, openCV 4.0
4. 글자넣기 putText()


## ?
1. circularity = 4*pi*area/(perimeter\^2)
- https://answers.opencv.org/question/59955/program-that-check-for-the-roundness-of-ellipse/

## 22.03.24 (THU)
1. Morphology(형태학) Operations
- 4 popular operations
  1. Erosion (침식)
  2. Dilation (팽창)
  3. Opening
  4. Closing
- other operations
  5. 
2. drawKeypoints
- [OpenCV Blob or Circle Detection](https://www.delftstack.com/howto/python/opencv-blob-detection/)
- 
3. 







