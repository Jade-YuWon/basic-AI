# basic-CV and Image Processing
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


## 22.03.28 (TUE)
1. kNN
2. Processing of CT Lung Images as a Part of Radiomics - Aleksandr G. Zotin et al.
- https://www.researchgate.net/publication/342118026_Processing_of_CT_Lung_Images_as_a_Part_of_Radiomics
- Segmentation - PCA: 주성분 분석(主成分分析, Principal component analysis; PCA)은 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다.
- 병리학 (Pathology)

## 22.03.30 (WED)
1. 이미지 연산 (Image Operation)
- 블렌딩
- 차영상 
  - cv2.subtract(object_image, background_image)
2. 이미지 반전 (Image Inversing)
- inversed_image = 255 - original_image
3. 임의의 이미지와 같은 크기의 검은 이미지를 만드려면?
4. 십자가 그리기
- drawMarker(InputOutputArray img, Point position, const Scalar& color, int markerType = MARKER_CROSS, int markerSize=20, int thickness=1, int line_type=8);
5. 컨투어의 중심좌표 얻기
- 모멘트 이용
- M = moments(contour)
- center_x = int(M\['m10']/M\['m00'])
- center_y = int(M\['m01']/M\['m00'])
- (center_x, center_y)
6. 이미지 분할하여 저장
- 이미지 가로 폭(너비), 세로 폭(높이) 구하기
  - img3_h, img3_w = img3.shape\[:2]
- 원하는 비율로 분할하여 저장
  - img3_w_half = int(img3_w / 2)
  - img3_A = img3\[0:img3_h, 0:img3_w_half]
  - img3_B = img3\[0:img3_h, img3_w_half:]
 
## 22.04.03 (SUN)
- Histogram(히스토그램)
  - 값이 높을 수록 밝고, 낮을 수록 어둡다.
  - 균등할수록 명암비가 높고 선명하며, 밀집되어있을수록 명암비가 낮고 흐릿하다.
    - Histogram Stretching, Normalazation (히스토그램 스트레칭, 정규화): 명암비 향상
    - Histogram Equalization(히스토그램 균등화, 평활화, 평탄화): 어두운 곳은 더 어둡게, 밝은 곳은 더 밝게 한다. (스트레칭보다 좀 더 나은 품질, 빈도 저고고저)
    - CLAHE(Contrast Limited Adaptive Histogram Equalization): 이미지 일부분에만 equalization을 적용하는 기법. (일정한 영역을 분리하여 해당 영역에 대한 히스토그램 균등화 연산을 수행해 그 결과를 조합)
    - Backprojection: 2차원 히스토그램을 응용하여 이미지에서 원하는 객체만을 추출해 내는 방법
  - 참고 http://www.gisdeveloper.co.kr/?p=6652 https://www.charlezz.com/?p=44834 https://deep-learning-study.tistory.com/122
## 22.04.05 (TUE)
- Multilinear subspace learning
  - 영어에서 번역됨-다중 선형 부분 공간 학습은 차원 축소에 대한 접근 방식입니다. 관측치가 벡터화되어 데이터 텐서로 구성되거나 관측치가 데이터 텐서에 연결된 행렬 인 데이터 텐서에서 차원 축소를 수행 할 수 있습니다. https://en.wikipedia.org/wiki/Multilinear_subspace_learning
  - 
## 22.04.06 (WED)
1. 이미지 리사이즈
- ![image](https://user-images.githubusercontent.com/77564623/162023192-ccf9d19e-0a74-453f-a550-2dd39b9ef9c5.png)
- cv2_imshow(cv2.resize(img3_color, (0, 0), fx=0.7, fy=0.7))
2. 형태학연산 cv2 코드
  - 1. Erosion (침식) 
    - eye_bin_erode = cv2.erode(eye_bin_close, kernel_eye, iterations=3)
  - 2. Dilation (팽창)
    - eye_dilated = cv2.dilate(eye_bin_erode, kernel_eye, iterations=3)
  - 3. Opening
  - 4. Closing
    - eye_bin_close = cv2.morphologyEx(bin3, cv2.MORPH_CLOSE, kernel_eye, iterations=1)

## 22.04.07 (THU)
- image binary operation https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
- Digital Twinization
  - 디지털 트윈(Digital Twin)은 물리적 자산, 시스템 또는 프로세스를 소프트웨어로 표현하는 것 https://www.gereports.kr/everthing-about-digital-twin/
  - 자산 디지털 트윈 (Asset Digital Twin)
  - 네트워크 디지털 트윈 (Network Digital Twin)
  - 프로세스 디지털 트윈 (Process Digital Twin)

## 22.04.14 (THU)
- Why are our eyes get saturated?
  - Answer: Our eye get too much of light signals by our rod cells, that is a kind of eye cells, that distinguish white and black. reference: ![image](https://user-images.githubusercontent.com/77564623/163228019-ea7d0c49-a694-4c9e-88fa-c97c23157938.png)
- astigmatism: can see lines from lights.

## 22.04.15 (FRI)
- AI EXPO KOREA 2022
  - GAN
    - 1. [Wikipedia: GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) : The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that is able to tell how much an input is "realistic", which itself is also being updated dynamically. This basically means that **the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator**. This enables the model to learn in an unsupervised manner.
    - 2. [Tensorflow Korea: Deep Convolutional GAN(심층 합성곱 생성적 적대 신경망) Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan?hl=ko) : generator(생성자) vs. discriminator(감별자) 
    - ![image](https://user-images.githubusercontent.com/77564623/163768870-9a617d99-0766-4089-b236-6a4938752d8a.png)
    - 3. [Arxiv Paper](https://arxiv.org/pdf/1406.2661.pdf) : [linearly interpolating](https://en.wikipedia.org/wiki/Linear_interpolation)
    - ![image](https://user-images.githubusercontent.com/77564623/163772890-7a7e84e0-f856-4cdb-b0fd-70a67cdb1860.png)
    - 4. [GAN 논문 리뷰 - WE GONNA MAKE IT](https://wegonnamakeit.tistory.com/54) : [Pytorch MNIST GAN Tutorial](https://ws-choi.github.io/blog-kor/seminar/tutorial/mnist/pytorch/gan/GAN-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC/), Minimax Problem of GAN, Backpropagation instead of MCMC(Markov Chain Monte Carlo)
    - 5. 응용(Application): [Img2Img](https://github.com/leehomyc/Img2Img-Translation-Networks), Super-resolution
    - 6. [NIPS 2016 GAN Tutorial](https://arxiv.org/pdf/1701.00160.pdf) : (1) Why generative modeling is a topic worth studying, (2) how generative models work, and how GANs compare to other generative models, (3) the details of how GANs work, (4) research frontiers in GANs, and (5) state-of-the-art image models that combine GANs with other methods. Finally, the tutorial contains three exercises for readers to complete, and the solutions to these exercises.
  - ViT
    - 1. [Transformer Wikipedia](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) --> [ViT Wikipedia](https://en.wikipedia.org/wiki/Vision_transformer)
    - 2. [[논문요약] Vision분야에서 드디어 Transformer가 등장 - ViT - KM-Hana](https://kmhana.tistory.com/27) : Transformers 종류 1. BERT : Self-Supervised Pre-Training Task / 2. GPT : 일방향 언어 모델링
    - ![image](https://user-images.githubusercontent.com/77564623/163774571-b5ee4bc3-269d-4418-b071-6c175da7a94a.png)
    - 3. CNN의 한계
    - ![image](https://user-images.githubusercontent.com/77564623/163774992-4a3eaa46-a9cc-4152-8426-906ae587ef0f.png)

## 22.05.06 (FRI)
- Image Crawling: Python Selenium, chromedriver library
- Python
  - try except
  - pass, continue, break

