/*单目标跟踪实验，分类器可以在几乎每一帧中检测出目标位置，在跟踪中，分类器只负责第一帧的检测*/
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/core/core.hpp>
# include <opencv2/objdetect.hpp>
# include "src/kcftracker.hpp"
# include <iostream>

using namespace std;
using namespace cv;

void detectAndDraw( Mat& frame, CascadeClassifier& cascade, double scale, bool tryflip );

String cascadeName = "haarcascade_frontalface_alt.xml";
int detected_ROI[4] = {0, 0, 0, 0};// 分别标记x1, y1, x2, y2
bool HOG = true;
bool FIXEDWINDOW = true;
bool MULTISCALE = true;
bool LAB = true;

class Queue
{
/*队列结构，用于绘制轨迹*/
private:
    const static int _Long = 30;
public:
    int Data[_Long][2];
    int Front;
    int rear;
    int num;
    Queue()
    {
        Front = 0;
        rear = 0;
        num = 0;
    }


    bool Empty()
    {
        if(Front == rear)
            return true;
        else
            return false;
    }

    void append(int _data[2])
    {
        Data[rear][0] = _data[0];
        Data[rear][1] = _data[1];
        rear++;
        num++;
        if(rear > _Long - 1)
            rear = 0;
        if(num > _Long)
        {
            Front++;
            num = _Long;
        }        
        if(Front > _Long - 1)
            Front = 0;
    }

    void Draw_Trace(Mat frame)
    {
        int index = Front + 1;
        int p_index = Front;
        if(num > 0)
        {
            for(int i = 1; i < num; i++)
            {
                if(index > _Long - 1)
                    index = 0;
                Point p1 = Point(Data[p_index][0], Data[p_index][1]);
                Point p2 = Point(Data[index][0], Data[index][1]);
                cv::line(frame, p1, p2, CV_RGB(255, 0, 0), 2);
                p_index = index;
                index ++;
            }
        }
    }
};

int main()
{
    bool tracking = false;
    bool detected = false;
    bool quit = false;
    Rect TrackWindow;
    Rect Tracker_ROI;
    Queue Trace;

    //分类器相关
    bool tryflip = false;
    CascadeClassifier cascade;
    double scale = 1;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    //视频读取相关
    VideoCapture capture(0);
    if( !capture.isOpened() )
    {
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        return -1;
    }
    
    Mat frame;                          //承载每一帧的图像
    namedWindow("Tracken");
    bool stop = false;                  //定义一个用来控制读取视频循环结束的变量
    
    //下面先读一帧图片用于初始化跟踪器
    capture.read(frame);
    // create the tracker
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    const static Scalar Track_color= CV_RGB(255, 0, 0);

    while(1)
    {
        //读取下一帧
        if(!capture.read(frame))
        {
            cout<<"Video Read failed"<<endl;
            return -1;  
        }

        if( !detected )
        {
            detectAndDraw(frame, cascade, scale, tryflip);

            int xMin, yMin, width, height;
            width = cvRound((detected_ROI[3] - detected_ROI[1]) * 1.2);
            height = cvRound((detected_ROI[2] - detected_ROI[0]) * 1.2);
            xMin = cvRound(detected_ROI[0] - (detected_ROI[2] - detected_ROI[0]) * 0.1);
            yMin = cvRound(detected_ROI[1] - (detected_ROI[3] - detected_ROI[1]) * 0.1);
            TrackWindow = Rect(xMin, yMin, width, height);
            tracker.init(Rect(xMin, yMin, width, height), frame);
        }


        if( tracking )
        {
            double t = (double)cvGetTickCount();

            Tracker_ROI = tracker.update(frame);

            int Center_x, Center_y;
            Center_x = Tracker_ROI.x + Tracker_ROI.height / 2;
            Center_y = Tracker_ROI.y + Tracker_ROI.width / 2;
            int Center[2] = {Center_x, Center_y};
            Trace.append(Center);

            rectangle(frame, Tracker_ROI, Track_color, 3, 8, 0);
            Trace.Draw_Trace(frame);

            t = ((double)cvGetTickCount() - t)/getTickFrequency();
            cout << "Track Speed: " << 1. / t << "fps" << endl;

        }     

        imshow("Tracken", frame);
        char c = waitKey(30);
        switch(c)
        {
            case 'q':
                quit = true;
            case 's':// start to detect
                tracking = true;
                detected = true;
            default:
                ;
        }
        if(quit) break;
    }
    //关闭视频文件
    capture.release();
    waitKey(0);
    return 0;
}


void detectAndDraw( Mat& frame, CascadeClassifier& cascade, double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors= CV_RGB(128,255,0);
    Mat gray, smallframe( cvRound (frame.rows/scale), cvRound(frame.cols/scale), CV_8UC1 );
    cvtColor( frame, gray, CV_BGR2GRAY );
    resize( gray, smallframe, smallframe.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallframe, smallframe );
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallframe, faces,
        1.1, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT
        |CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallframe, smallframe, 1);
        cascade.detectMultiScale( smallframe, faces2,
                                 1.1, 2, 0
                                 |CV_HAAR_FIND_BIGGEST_OBJECT
                                 |CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
            faces.push_back(cvRect(smallframe.cols - r->x - r->width, r->y, r->width, r->height));
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    //Draw
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
    {
        Mat smallframeROI;
        vector<Rect> nestedObjects;
        Point center;
        int radius;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            Mat srcROI=frame(cvRect(r->x,r->y,r->width,r->height));
            rectangle( frame,cvPoint(center.x-radius,center.y-radius),cvPoint(center.x+radius,center.y+radius),colors, 3, 8, 0);
        }
        else
            rectangle( frame, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       colors, 3, 8, 0);
        detected_ROI[0] = cvRound(r->x*scale);
        detected_ROI[1] = cvRound(r->y*scale);
        detected_ROI[2] = cvRound((r->x + r->width-1)*scale);
        detected_ROI[3] = cvRound((r->y + r->height-1)*scale);
        // cout << *detected_ROI << detected_ROI[1]  <<  detected_ROI[2] << detected_ROI[3] << endl;
    }
}