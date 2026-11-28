#ifndef FT_DETECTOR_ARMOR_DETECTOR_NN_H
#define FT_DETECTOR_ARMOR_DETECTOR_NN_H

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <deque>
#include <string>
#include <vector>

#include "ft_detector/armor_detector.h"
#include "ft_detector/onnx_engine.h"

namespace ft_nn
{
//解析模型路径
std::string resolveModelPath(const std::string& path);
// letterbox 参数：记录缩放和补边信息，供坐标还原使用
struct LetterboxInfo
{
    float scale = 1.0f;//缩放比例
    int pad_x = 0;//左右两侧的补边宽度
    int pad_y = 0;//上下两侧的补边高度
    cv::Size canvas{640,480};//最终画布尺寸
    //将画布上的坐标映射回原图
    cv::Point2f toSource(const cv::Point2f& p) const
    {
        return cv::Point2f((p.x - static_cast<float>(pad_x)) / scale,
                           (p.y - static_cast<float>(pad_y)) / scale);
    }
};
}

class ArmorDetectorNN
{
public:
    struct Params
    {
        //模型预处理
        std::string model_path;
        int input_w = 640;
        int input_h = 480;
        int pad_value = 124;//补边填充值（灰色）
        bool to_rgb = true;//是否将 BGR 转为 RGB
        //去重
        float color_thresh = 0.50f;// 颜色置信度阈值
        float number_thresh = 0.50f;// 编号置信度阈值
        float nms_iou = 0.50f;// NMS 去重阈值
        int max_detections = 8;
        //编号
        std::vector<int> large_by_number{0,0,0,0,0,0,0,1,1};
        int intra_threads = 2;
        bool verbose_output = true;// 是否打印输出张量的统计信息
        int debug_every_n = 30;// 每 N 帧打印一次耗时
    };
    
    bool init(const Params& params);
    bool initFromRos(const ros::NodeHandle& nh);// 从 ROS 参数服务器初始化
    bool ready() const {return ready_;};

    //核心检测接口：输入 BGR 图像，输出检测结果
    std::vector<Armor> detect(const cv::Mat& bgr);
    //可视化绘制接口
    void drawArmors(cv::Mat& image, const std::vector<Armor>& armors) const;
    // 获取上一次 Letterbox 的参数，供外部调试
    const ft_nn::LetterboxInfo& lastLetterbox() const {return last_lb_;}
    // 获取上一次的总耗时
    double lastTotalMs() const {return last_total_ms_;}
    // 获取当前使用的参数
    const Params& params() const {return params_;}
    
private:
    // 预处理：缩放、补边、归一化、转 NCHW
    cv::Mat preprocess(const cv::Mat& bgr, ft_nn::LetterboxInfo& lb) const;
    // 后处理：解析输出张量，过滤置信度并还原坐标
    std::vector<Armor> decode(const cv::Mat& output,
                              const ft_nn::LetterboxInfo& lb,
                              const cv::Size& src_szie) const;
    // NMS 非极大值抑制去重
    static void nms(std::vector<Armor>& armors, float iou_thresh);
    // 调试辅助函数：打印输出张量统计信息、打印耗时
    void logOutoutStats(const cv::Mat& ouput) const;
    void logTiming();
    
    OnnxEngine engine_;
    Params params_;
    bool ready_ = false;
    ft_nn::LetterboxInfo last_lb_;
    
    double last_pre_ms_ = 0.0;//预处理耗时
    double last_post_ms_ = 0.0;//后处理耗时
    double last_total_ms_ = 0.0;//总耗时
    std::deque<double> total_hits_;//记录最近 100 次的耗时，用于计算 avg 和 p95
    int frame_count_ = 0;// 帧计数器
};

#endif