#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
void drawDottedContour(Mat& frame, const vector<Point>& contour,
                      const Scalar& color = Scalar(0, 255, 0), int dotSpacing = 10) {
    for (int i = 0; i < contour.size(); i += dotSpacing) {
        Point point = contour[i];
        circle(frame, point, 2, color, -1);
    }
}
Mat applyCLAHEBGR(const Mat& image) {
    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);
    vector<Mat> labChannels;
    split(lab, labChannels);
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    Mat cl;
    clahe->apply(labChannels[0], cl);
    labChannels[0] = cl;
    Mat limg;
    merge(labChannels, limg);
    Mat result;
    cvtColor(limg, result, COLOR_Lab2BGR);
    return result;
}
int main() {
    string videoPath = "video.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cout << "❌ Failed to open video." << endl;
        return -1;
    }
    double fps = cap.get(CAP_PROP_FPS);
    int delay = (fps > 0) ? static_cast<int>(1000 / fps) : 33;

    Mat frame;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }
        resize(frame, frame, Size(640, 360));

        Mat enhancedFrame = applyCLAHEBGR(frame);

        Mat hsv;
        cvtColor(enhancedFrame, hsv, COLOR_BGR2HSV);

        Scalar lowerWhite(0, 0, 180);
        Scalar upperWhite(180, 30, 255);
        Mat mask;
        inRange(hsv, lowerWhite, upperWhite, mask);

        int height = mask.rows;
        int width = mask.cols;

        mask(Rect(0, 0, width, static_cast<int>(height * 0.4))).setTo(0);

        mask(Rect(0, static_cast<int>(height * 0.70), width,
                 height - static_cast<int>(height * 0.70))).setTo(0);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 1));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (contourArea(contour) > 150) {
                Rect boundingBox = boundingRect(contour);
                int x = boundingBox.x;

                if (x > width * 0.3 && x < width * 0.7) {
                    continue;
                }

                drawDottedContour(frame, contour);
            }
        }
        imshow("Lane Detection", frame);
        imshow("Lane Detection - Mask", mask);

        if (waitKey(delay) == 'q') {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
