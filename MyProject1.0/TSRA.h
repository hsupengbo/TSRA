//
// Created by XuPengbo on 2020/11/1.
//

#ifndef MYPROJECT_TSRA_H
#define MYPROJECT_TSRA_H
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

struct sign{
    cv::Mat image;
    cv::Point2i P1;
    cv::Point2i P2;
    string label;
};
void fillHole(const Mat srcBw, Mat& dstBw);
bool isCircle(const Mat srcBw, Mat& mytemp);
double sigmoid(int num, int s);
void sigmoid_constract(Mat& image, Mat& outimage, int s);
void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity);

void FindColorSign(int color_option,Mat input,std::vector<sign>& signs);
void Find_Traffic_Sign(cv::String InputFolderPath,
                       cv::String OutputFolderPath);

#endif //MYPROJECT_TSRA_H
