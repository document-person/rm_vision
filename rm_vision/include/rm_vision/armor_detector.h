#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>//动态数组
#include <string>

struct LightBar
{
    cv::RotatedRect rect;//灯条的最小外接旋转矩形（包含中心点、宽高、倾斜角度）
    double angle = 0.0;//灯条角度
    cv::Point2f center;//中心点坐标
    double width = 0.0;//灯条宽度
    double length = 0.0;//灯条长度
    double mean_v = 0.0;//灯条区域的平均亮度(V)，用来区分"真灯条"和"地面反光"
};

struct Armor
{
    cv::Rect bounding_box;//外框
    std::string type;//大还是小
    std::string color;//红还是蓝
    std::vector<cv::Point2f> corners; // 四角点 (左上、右上、右下、左下) 供 PnP 使用
};

class ArmorDetector{
public:
    ArmorDetector();
    // 核心对外接口：输入一张彩色图，输出识别到的所有装甲板信息
    std::vector<Armor> detect(const cv::Mat& bgr_image);
    //画框
    void drawArmors(cv::Mat& image,const std::vector<Armor>& armors);

private:
    //颜色分割
    cv::Mat getColorMask(const cv::Mat& hsv,bool is_blue);
    //找灯条
    std::vector<LightBar> findLightBars(const cv::Mat& mask,const cv::Mat& hsv);
    //配对装甲板
    std::vector<Armor> matchArmors(
        const std::vector<LightBar>& bars,const std::string& color);
};
#endif
