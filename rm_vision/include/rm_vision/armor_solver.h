#ifndef ARMOR_SOLVER_H
#define ARMOR_SOLVER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <ros/time.h>
#include <memory>
#include <std_msgs/Header.h>
#include <geometry_msgs/TwistStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include "rm_vision/armor_detector.h"

class ArmorSolver
{
public:
    struct Pose
    {
        cv::Vec3d rvec{0,0,0};//旋转向量 
        cv::Vec3d tvec{0,0,0};//平移，单位米，相机系
        cv::Matx33d R = cv::Matx33d::eye();//旋转矩阵
        tf2::Quaternion q{0,0,0,1};//四元数
        double reproj_rms = 0.0;//重投影（算精修后的误差）
        bool valid = false;
    };
    //参数服务器接口
    struct Params
    {
        float small_half_w = 0.0675f;
        float big_half_w = 0.1175f;
        float bar_half_len = 0.0275f;
    };
    void setParams(const Params& params){
        params_ = params;
    }
    ArmorSolver();
    //接收相机内参
    void setCameraInfo(const cv::Mat& K,const cv::Mat& D,const cv::Size& image_size);
    //判断内参是否可用
    bool ready(void) const{return ready_;}
    //解算3D位姿和发布TF
    bool solve(const Armor& armor,const cv::Size& image_size,Pose& out);
    //重投影误差计算
    double reprojectionRms(const std::vector<cv::Point3f>& obj,
                           const std::vector<cv::Point2f>& corner,
                           const cv::Vec3d& rvec, const cv::Vec3d& tvec) const;
    //画出坐标系
    void drawAxes(cv::Mat& image,const Pose& pose) const;
    void broadcastTF(const Pose& pose,const std_msgs::Header& header,
                     const std::string& child_frame);
    //TF广播
    void setTfParent(const std::string& frame){
        tf_parent_ = frame;
    }
private:
    //确定灯条四个点的3D坐标
    std::vector<cv::Point3f> find3DPoints(const std::string& type);
    //检查角点顺序
    bool cornerscheck(const std::vector<cv::Point2f>& corners);
    cv::Mat K_;//CV_64F  3x3
    cv::Mat D_;//CV_64F  1xN
    cv::Size image_size_;//CameraInfo 声明的分辨率
    bool ready_ = false;
    bool has_last_ = false;
    cv::Vec3d last_nz_{0.0,0.0,1.0};//上一帧板法向量
    ros::Time last_time_;
    // //solve函数最后的调试用：整段数据里见过的最大的 w_px/h_px（板子最正对的那一帧）
    // double max_aspect_ = 0.0;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string tf_parent_;//TF 父坐标系（空 = 用 header.frame_id）
    Params params_;
};

#endif
