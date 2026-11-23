#ifndef FT_TRACKER_TRACKER_H
#define FT_TRACKER_TRACKER_H

#include <vector>
#include <opencv2/core.hpp>
#include <cstdint>
#include <opencv2/video/tracking.hpp>

namespace ft_tracker{

//存单帧观测数据
struct Measurement
{
    cv::Point2f center{0.f, 0.f}; //四角点像素中心，关联用
    double width = 0.0;//板宽，用做板宽归一化
    double pos[3] = {0.0, 0.0, 0.0};//中心坐标，相机系
    double q[4] = {0.0, 0.0, 0.0, 1.0};
    bool is_red = false;
    bool is_large = false;
};
//单条轨迹
struct TrackState
{
    uint32_t id = 0;//跨帧唯一id
    cv::Point2f center{0.f, 0.f};//图像像素中心
    double width = 0.0;//板宽
    double pos[3] = {0.0, 0.0, 0.0};//滤波后的中心坐标
    double q[4] = {0.0, 0.0, 0.0, 1.0};
    double vel[3] = {0.0, 0.0, 0.0};//速度约定x,y,z
    double predict_pos[3] = {0.0, 0.0, 0.0};//时刻的预测位置
    double sigma_v = 0.0;//速度标准差   
    bool is_red = false;
    bool is_large = false;
    bool tracking = false;
    bool confirmed = false;// 已连续命中足够帧数
    int hits = 0;//累计命中帧数，节点层用它选"主目标"
    double lost_time = 0.0;//丢失时间
    bool valid = false;//速度是否可信
};

class Tracker
{
public:
    struct Params
    {
        double gate = 11.34;//用马氏距离后，z 噪声大时 z 方向的门限会自动放宽，x/y 会自动收紧
        double gate_sigma_m = 0.05;//关联门限实际下限
        int confirm_hits = 2;// 连续命中几帧才对外发布，压掉单帧虚检
        double lost_time = 0.5;//多久没观测就删轨迹
        int max_tracks = 8;//轨迹数上限

        double R_xy = 1e-4;//XY 测量方差
        double R_z = 0.02;//Z 测量方差
        double acc_var = 6.0;//加速度方差
        double vel_var_max = 0.5;//速度方差上限
        double v_max = 8.0;//限速
        double v0_var = 1.0;//新轨迹的速度方差初值
        double a_max = 10.0;//单帧最大加速度
        double dv_step = 0.5;//单帧速度修正的硬上限
        double T = 0.1;

        double g_reacq_time = 1.5;//目标层 ID：允许断档的最长时间
        double g_reacq_dist = 0.8;//目标层 ID：允许跳变的最大距离
        double g_id_memory  = 4.0;//断档后还能保留 id 的最长时间，比输出保持更长
    };
    
    void setParams(const Params& p){params_ = p;}
    //每帧调用，传入观测数组和时间戳，返回确认的轨迹
    std::vector<TrackState> update(const std::vector<Measurement>& meas,double stamp);
    void reset();
    //累计建过多少条轨迹；用来判断是"关联失败"还是"目标真的多"
    size_t totalCreated() const { return static_cast<size_t>(next_id - 1); }

private:
    //内部轨迹
    struct Track
    {
        uint32_t id = 0;//内部id
        Measurement last;//最近一次观测信息
        double last_stamp = 0.0;
        int hits = 0;
        int misses = 0;//连续丢失帧数，本帧配上了就清零
        bool confirmed = false;

        cv::KalmanFilter kf;
        bool kf_init_ready = false;
    };
    
    void initKalman(Track& tr, const Measurement& m);
    void predictKalman(Track& tr, double dt);//计算Q和F并预测
    void clampVelocity(cv::Mat& state) const;//速度限幅，防止单帧异常把状态带偏
    void clampVelocityDelta(cv::Mat& post, const cv::Mat& pre, double dt) const;//单帧速度变化限幅
    
    std::vector<Track> tracks_;
    std::vector<TrackState> last_out_;//缓存上一帧，防时间回跳
    uint32_t next_id = 1;
    double last_stamp_ = 0.0;
    Params params_;
};

}
#endif