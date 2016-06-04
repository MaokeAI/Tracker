# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc/imgproc.hpp>
# include <opencv2/video/tracking.hpp>
# include <opencv2/core/core.hpp>
# include <opencv2/objdetect.hpp>
# include <iostream>

using namespace std;
using namespace cv;

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, bool tryflip );

String cascadeName = "haarcascade_frontalface_alt.xml";
int detected_ROI[4] = {0, 0, 0, 0};// 分别标记x1, y1, x2, y2

class CamShiftTracking
{
private:
    Mat hist;
    Mat mask;
    int hsize;
    float hranges[2];

public:
    CamShiftTracking()
    {

    }

    CamShiftTracking(Mat image,int vmin,int vmax,int smin, int smax, Rect selection)
    {
        Mat hsv, hue;
        getHue(image, hsv, hue);
        inRange(hsv, Scalar(0, smin, vmin),Scalar(180, smax, vmax), mask);        
        Mat roi(hue, selection), maskroi(mask, selection);
        hsize = 16;
        hranges[0] = 0;
        hranges[1] = 180;
        const float* phranges = hranges;
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, CV_MINMAX);
    }

    Rect track(Mat image, Rect &bb)
    {
        Mat hue;
        getHue(image,hue);
        Mat backproj;
        const float* phranges = hranges;
        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
        backproj &= mask;
        RotatedRect trackBox = CamShift(backproj, bb,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        return trackBox.boundingRect();
    }

    Rect meanShiftTrack(Mat image,Rect &bb)
    {
        Mat hue;
        getHue(image,hue);
        Mat backproj;
        const float* phranges = hranges;
        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
        backproj &= mask;
        meanShift(backproj, bb, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        return bb;
    }

    void getHue(const Mat &img,Mat &hsv, Mat &hue)
    {
        cvtColor(img, hsv, COLOR_BGR2HSV);
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());//h 0-180 s v 0-255
        mixChannels(&hsv, 1, &hue, 1, ch, 1);
    }

    void getHue(const Mat &img, Mat &hue)
    {
        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());//h 0-180 s v 0-255
        mixChannels(&hsv, 1, &hue, 1, ch, 1);
    }
};

int main()
{
    Mat img;
    VideoCapture cap;
    Rect TrackWindow;
    CamShiftTracking trackor;
    bool tracking = false;
    bool detected = false;
    bool quit = false;

    //分类器相关
    bool tryflip = false;
    CascadeClassifier cascade;
    double scale = 1;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }
    
    cap.open(0);
    if( !cap.isOpened() )
    {
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        return -1;
    }
    namedWindow("CamShift");

    while(1)
    {
        //读取下一帧
        if(!cap.read(img))
        {
            cout<<"Video Read failed"<<endl;
            return -1;  
        }

        if( !detected )
        {
            detectAndDraw(img, cascade, scale, tryflip);

            int xMin, yMin, width, height;
            width = cvRound((detected_ROI[3] - detected_ROI[1]) * 1.3);
            height = cvRound((detected_ROI[2] - detected_ROI[0]) * 1.3);
            xMin = cvRound(detected_ROI[0] - (detected_ROI[2] - detected_ROI[0]) * 0.15);
            yMin = cvRound(detected_ROI[1] - (detected_ROI[3] - detected_ROI[1]) * 0.15);
            TrackWindow = Rect(xMin, yMin, width, height);

            trackor = CamShiftTracking(img, 32, 255, 60, 200, TrackWindow);
        }

        if( tracking )
        {
            double t = (double)cvGetTickCount();
            Rect bb = trackor.track(img, TrackWindow);
            rectangle(img, bb, Scalar(0, 0, 255), 3, CV_AA);

            t = ((double)cvGetTickCount() - t)/getTickFrequency();
            cout << "Track Speed: " << 1. / t << "fps" << endl;

        }

        imshow("CamShift", img);
        char c = waitKey(30);
        switch(c)
        {
            case 'q':
                quit = true;
            case 't':// start to detect
                tracking = true;
                detected = true;
            default:
                ;
        }

        if(quit) break;
    }
    return 0;
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors= CV_RGB(128,255,0);
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT
        |CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 |CV_HAAR_FIND_BIGGEST_OBJECT
                                 |CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
            faces.push_back(cvRect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
    }
    t = ((double)cvGetTickCount() - t)/getTickFrequency();
    cout << "Detect Speed: " << 1. / t << "fps" << endl;

    //Draw
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        int radius;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            Mat srcROI=img(cvRect(r->x,r->y,r->width,r->height));
            rectangle( img,cvPoint(center.x-radius,center.y-radius),cvPoint(center.x+radius,center.y+radius),colors, 3, 8, 0);
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       colors, 3, 8, 0);
        detected_ROI[0] = cvRound(r->x*scale);
        detected_ROI[1] = cvRound(r->y*scale);
        detected_ROI[2] = cvRound((r->x + r->width-1)*scale);
        detected_ROI[3] = cvRound((r->y + r->height-1)*scale);
        // cout << *detected_ROI << detected_ROI[1]  <<  detected_ROI[2] << detected_ROI[3] << endl;
    }
}