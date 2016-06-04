# KCF Face Tracker
It is a tracker using KCF and OpenCV
## based OpenCV 
首先由Haar特征的Adaboost分类器检测人脸，然后可以通过KCF算法进行人脸跟踪
这里调用的KCF源码由https://github.com/joaofaro/KCFcpp下载
## compile
g++ classTest.cpp -o camTracker `pkg-config --libs opencv --cflags`

## run
./Video

## Usage
1. press 's' to start face tracking
2. press 'q' to quit 

