# Camshift Face Tracker
It is a tracker using Camshift based on OpenCV
## Method
首先由Haar特征的Adaboost分类器检测人脸，然后可以通过Camshift算法进行人脸跟踪

## compile
g++ classTest.cpp -o camTracker `pkg-config --libs opencv --cflags`

## run
./Video

## Usage
1. press 's' to start face tracking
2. press 'q' to quit 

