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
    //参数服务器接口
    struct Params
    {
        int red_h_min1 = 0, red_h_max1 = 20, red_h_min2 = 150, red_h_max2 = 180;
        int blue_h_min = 95, blue_h_max = 135;
        int s_min = 90, s_max = 255, v_min = 120, v_max = 255;
        
        double min_area = 20.0;
        double min_bar_ratio = 1.5, max_bar_ratio = 20.0;
        double max_bar_angle = 45.0;
        double min_fill_ratio = 0.5;
        
        double max_y_diff = 0.25, max_angle_diff = 6.0;
        double max_bar_length_diff = 0.20, max_bar_width_diff = 0.22;
        double min_v_control = 0.93;
        double max_x_dist_ratio = 6.0, min_x_dist_ratio = 1.0;
        double bar_size_ratio = 3.2;
    };
    void setParams(const Params& params){
        params_ = params;
    }

private:
    //颜色分割
    cv::Mat getColorMask(const cv::Mat& hsv,bool is_blue);
    //找灯条
    std::vector<LightBar>findLightBars(const cv::Mat& mask,const cv::Mat& hsv);
    //配对装甲板
    std::vector<Armor> matchArmors(
        const std::vector<LightBar>& bars,const std::string& color);
    //参数服务器
    Params params_;
};
#endif
