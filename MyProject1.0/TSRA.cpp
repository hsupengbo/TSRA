//
// Created by XuPengbo on 2020/11/1.
//
#include "TSRA.h"
double PI = 3.1415926535;
RNG rng(12345);
string ths[5] = { "1","2","3","4","5" };
string label[3] = { "r","y","b" };
void fillHole(const Mat srcBw, Mat& dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));//填充区域

    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

bool isCircle(const Mat srcBw, Mat& mytemp)
{//输入的是一个灰度图像
    Mat temp = Mat::zeros(srcBw.size(), CV_8UC1);
    bool iscircle = false;
    //获得srcBw信息
    int w = srcBw.cols;
    int h = srcBw.rows;
    int w1 = mytemp.cols;
    int h1 = mytemp.rows;
    //cout << w << " " << w1 << " " << h << " " << h1 << endl;
    int count1 = 0;//各部分的缺失像素计数器
    int count2 = 0;
    int count3 = 0;
    int count4 = 0;
    //将srcBw平均分成四份,进行访问缺失的像素个数、所占比重
    //先访问左上
    for (int i = 0;i < h / 2;i++)
    {
        for (int j = 0;j < w / 2;j++)
        {
            if (srcBw.at<uchar>(i, j) == 0)
            {
                temp.at<uchar>(i, j) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 0) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 1) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 2) = 255;
                count1++;
            }
        }
    }
    //右上
    for (int i = 0;i < h / 2;i++)
    {
        for (int j = w / 2 - 1;j < w;j++)
        {
            if (srcBw.at<uchar>(i, j) == 0)
            {
                temp.at<uchar>(i, j) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 0) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 1) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 2) = 255;
                count2++;
            }
        }
    }
    //左下
    for (int i = h / 2 - 1;i < h;i++)
    {
        for (int j = 0;j < w / 2;j++)
        {
            if (srcBw.at<uchar>(i, j) == 0)
            {
                temp.at<uchar>(i, j) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 0) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 1) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 2) = 255;
                count3++;
            }
        }
    }
    //右下
    for (int i = h / 2 - 1;i < h;i++)
    {
        for (int j = w / 2 - 1;j < w;j++)
        {
            if (srcBw.at<uchar>(i, j) == 0)
            {
                temp.at<uchar>(i, j) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 0) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 1) = 255;
                mytemp.at<uchar>(i, j * mytemp.channels() + 2) = 255;
                count4++;
            }
        }
    }
    float c1 = (float)count1 / (float)(w * h);//左上
    float c2 = (float)count2 / (float)(w * h);//右上
    float c3 = (float)count3 / (float)(w * h);//左下
    float c4 = (float)count4 / (float)(w * h);//右下
    //imshow("temp",mytemp);
    //cout << "result: " << c1 << "," << c2
    //<< "," << c3 << "," << c4 << endl;

    //限定每个比率的范围
    if ((c1 > 0.037 && c1 < 0.12) && (c2 > 0.037 && c2 < 0.12) && (c3 > 0.037 && c3 < 0.12) && (c4 > 0.037 && c4 < 0.12))
    {
        //限制差值,差值比较容错，相邻块之间差值相近，如左上=右上&&左下=右下或左上=左下&&右上=右下
        if ((abs(c1 - c2) < 0.04 && abs(c3 - c4) < 0.04) || (abs(c1 - c3) < 0.04 && abs(c2 - c4) < 0.04))
        {
            iscircle = true;
        }
    }


    return iscircle;
}

double sigmoid(int num, int s)
{
    double result = 1 / (1 + pow(2.7, (127.5 - num) / s));
    result *= 1.3;
    return min(1.0, result);
}

void sigmoid_constract(Mat& image, Mat& outimage, int s)
{
    const uchar* input = image.data;
    uchar* output = outimage.data;
    for (int i = 0;i < image.rows;i++)
    {
        for (int j = 0;j < image.cols;j++)
        {
            for (int th = 0;th < 3;th++)
            {
                output[i * outimage.step + j * 3 + th] = round(sigmoid(input[i * image.step + j * 3 + th], s) * 255);
            }
        }
    }

}

void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{
    double r, g, b;double h, s, i;double sum;
    double minRGB, maxRGB;
    double theta;
    r = red / 255.0;
    g = green / 255.0;
    b = blue / 255.0;
    minRGB = ((r < g) ? (r) : (g));
    minRGB = (minRGB < b) ? (minRGB) : (b);
    maxRGB = ((r > g) ? (r) : (g));
    maxRGB = (maxRGB > b) ? (maxRGB) : (b);
    sum = r + g + b;
    i = sum / 3.0;
    if (i < 0.001 || maxRGB - minRGB < 0.001)
    {
        h = 0.0;s = 0.0;
    }
    else
    {
        s = 1.0 - 3.0 * minRGB / sum;
        theta = sqrt((r - g) * (r - g) + (r - b) * (g - b));
        theta = acos((r - g + r - b) * 0.5 / theta);
        if (b <= g) h = theta;
        else h = 2 * PI - theta;
        if (s <= 0.01) h = 0;
    }
    hue = (int)(h * 180 / PI);
    saturation = (int)(s * 100);
    intensity = (int)(i * 100);
}

void FindColorSign(int coloroption,cv::Mat input,std::vector<sign>& signs){
    /*
    * Input: 颜色检测选项:(int)color_option,检测图片(Mat)input，检测到的图标信息vector<sign> Signs
    * Output: 输出检测到的的vector<sign>
    */
    Mat src_in = input;
    Mat copy;
    src_in.copyTo(copy);
    Mat src(src_in.rows, src_in.cols, CV_8UC3);
    //src = src_in;
    sigmoid_constract(src_in, src, 50);
    int width = src.cols;
    int height = src.rows;
    double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
    Mat Mat_rgb = Mat::zeros(src.size(), CV_8UC1);
    int x, y;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {

            B = src.at<Vec3b>(y, x)[0];
            G = src.at<Vec3b>(y, x)[1];
            R = src.at<Vec3b>(y, x)[2];
            RGB2HSV(R, G, B, H, S, V);
            //red
            if (coloroption == 0 && (H >= 340 || H <= 11) &&
                S >= 6 && S <= 70 && V > 15 && V <= 85)
            {
                Mat_rgb.at<uchar>(y, x) = 255;
            }
            //yellow
            if (coloroption == 1 && H >= 15 && H <= 50 &&
                S >= 35 && S <= 155 && V > 25 && V <= 100)
            {
                Mat_rgb.at<uchar>(y, x) = 255;
            }
            //blue
            if (coloroption == 2 && H >= 100 && H <= 230 &&
                S >= 35 && S <= 155 && V > 25 && V <= 155)
            {
                Mat_rgb.at<uchar>(y, x) = 255;
            }
        }
    }
    medianBlur(Mat_rgb, Mat_rgb, 3);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
    Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(Mat_rgb, Mat_rgb, element);
    dilate(Mat_rgb, Mat_rgb, element1);
    fillHole(Mat_rgb, Mat_rgb);
    Mat Mat_rgb_copy = Mat_rgb;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(Mat_rgb, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f>center(contours.size());
    vector<float>radius(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
        minEnclosingCircle(contours_poly[i], center[i], radius[i]);
    }
    Mat drawing = Mat::zeros(Mat_rgb.size(), CV_8UC3);
    int count1 = 0;
    int th = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        Rect rect = boundRect[i];
        //首先进行一定的限制，筛选出区域
        //高宽比限制
        float ratio = (float)rect.width / (float)rect.height;
        //轮廓面积
        float Area = (float)rect.width * (float)rect.height;
        float dConArea = (float)contourArea(contours[i]);
        float dConLen = (float)arcLength(contours[i], 1);
        if (dConArea < 800 )
            continue;
        if (ratio >= 1.8 || ratio <= 0.6)
            continue;

        //进行圆筛选，通过四块的缺失像素比较
        Mat roiImage;
        Mat_rgb_copy(rect).copyTo(roiImage);
        Mat temp,temp1;
        copy(rect).copyTo(temp);

        //imshow("test2",temp);
        bool iscircle = isCircle(roiImage, temp);
        cout << "circle:" << iscircle << endl;
        if (!iscircle && coloroption==0) continue;
        //接下来是形状限制！这里检测圆形标志牌
        float C = (4 * PI * dConArea) / (dConLen * dConLen);
        //if (C < 0.4 && coloroption == 0)	continue;
        //imshow("test2", temp);
        //string name = "test".append(ths[th]);
        //imshow("test", temp);
        input(rect).copyTo(temp1);
        sign new_sign;
        new_sign.image = temp1;
        new_sign.label = label[coloroption];
        new_sign.P1 = cv::Point2i(rect.x, rect.y);
        new_sign.P2 = cv::Point2i(rect.x + rect.width, rect.y + rect.height);
        signs.push_back(new_sign);
        copy(rect).copyTo(roiImage);

        Mat temp2 = Mat::zeros(temp.size(), CV_8UC1);
        cvtColor(temp, temp2, COLOR_BGR2GRAY);
        resize(temp2, temp2, Size(45, 45));

        temp2 = temp2.reshape(0, 1);
        temp2.convertTo(temp2, CV_32F);

        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        //putText(src, labelname[result], cvPoint(boundRect[i].x, boundRect[i].y - 10), 1, 1, CV_RGB(255, 0, 0), 2);//红色字体注释
        //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
        count1++;
        th++;
    }
    waitKey(0);

}
//调用示例:Find_Traffic_Sign("D:\\images\\test","D:\\images\\out");
void Find_Traffic_Sign(cv::String InputFolderPath, cv::String OutputFolderPath) {
    ofstream outfile(OutputFolderPath+"/out.txt");
    cv::String ImagesName= InputFolderPath + "/*.png";
    std::vector<cv::String> files;
    std::vector<sign> signs;
    cv::glob(ImagesName, files);
    size_t counts = files.size();
    cout<<"size="<<counts<<endl;
    cv::String saveFile, tmp1,tmp2,str;
    for (size_t i = 0; i < counts; ++i) {
        Mat tmp=cv::imread(files[i]);
        for(int color_option=0;color_option<3;++color_option){//颜色检测选项
            signs.clear();  FindColorSign(color_option,tmp,signs);
            cout <<"size:"<< signs.size() << endl;
            for(int j=0;j<signs.size();++j){
                str=to_string(j);
                tmp1=files[i].substr(InputFolderPath.length());   size_t le=tmp1.length();
                tmp2=tmp1.substr(0,le-4);   saveFile=OutputFolderPath+tmp2+"_"+str+".png";
                cout<<"outimgpath="<<saveFile<<endl;
                cv::imwrite(saveFile,signs[j].image);//写入图片,实际输出如: D:\images\out\10021_0.png
                outfile<<files[i]<<" "<<signs[j].P1.x
                       <<" "<<signs[j].P1.y
                       <<" "<<signs[j].P2.x
                       <<" "<<signs[j].P2.y
                       <<" "<<signs[j].label
                       <<endl;//写入信息
            }
        }
    }
}
