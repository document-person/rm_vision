#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include "rm_vision/armor_detector.h"
#include "rm_vision/armor_solver.h"
#include <ft_detection/DetectionFrame.h>
#include <rm_vision/ArmorParamsConfig.h>
#include <dynamic_reconfigure/server.h>

//发布解码后的图像
ros::Publisher debug_image_pub;
//创建检测消息发布
ros::Publisher detection_pub;
//定义一个全局变量，保存当前的参数状态
rm_vision::ArmorParamsConfig current_params;
//调用寻找装甲板这个类
ArmorDetector detector;
//调用装甲板处理
ArmorSolver solver;
//获取最近的时间戳
ros::Time last_image_stamp;
std::string last_frame_id;
//获取最近的相机内参信息
sensor_msgs::CameraInfo last_camera_info;
//用于后续判断是否收到相机内参
bool camera_info_received = false;
//提取K和D
cv::Mat last_camera_K;//3x3矩阵
cv::Mat last_dist_D;//畸变系数
int last_image_width = 0;
int last_image_height = 0;

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
void imageCallback(const sensor_msgs::ImageConstPtr& msg);
void reconfigCallback(rm_vision::ArmorParamsConfig &config,uint32_t level);

int main(int argc,char* argv[])
{
    ros::init(argc,argv,"image_sub");//节点名字为image_sub
    ros::NodeHandle nh("~");//用私有句柄

    //从参数服务器读取话题名
    std::string image_topic, camera_info_topic;
    nh.param<std::string>("image_topic",image_topic,"/hk_camera/image_raw");
    nh.param<std::string>("camera_info_topic",camera_info_topic,"/hk_camera/camera_info");
    // TF 父坐标系：默认 camera_optical_frame（机器人 TF 树上有这一帧）
    std::string tf_parent;
    nh.param<std::string>("tf_parent_frame", tf_parent, "camera_optical_frame");                                            
    solver.setTfParent(tf_parent);
    
    // 颜色阈值(HSV)
    nh.param("red_h_min1", current_params.red_h_min1, 0);
    nh.param("red_h_max1", current_params.red_h_max1, 20);
    nh.param("red_h_min2", current_params.red_h_min2, 150);
    nh.param("red_h_max2", current_params.red_h_max2, 180);
    nh.param("blue_h_min", current_params.blue_h_min, 95);
    nh.param("blue_h_max", current_params.blue_h_max, 135);
    nh.param("s_min", current_params.s_min, 90);
    nh.param("s_max", current_params.s_max, 255);
    nh.param("v_min", current_params.v_min, 120);
    nh.param("v_max", current_params.v_max, 255);

    // 灯条过滤
    nh.param("min_area", current_params.min_area, 20.0);
    nh.param("min_bar_ratio", current_params.min_bar_ratio, 1.5);
    nh.param("max_bar_ratio", current_params.max_bar_ratio, 20.0);

    // 装甲板物理尺寸
    nh.param("small_half_w", current_params.small_half_w, 0.0675);
    nh.param("big_half_w", current_params.big_half_w, 0.1175);
    nh.param("bar_half_len", current_params.bar_half_len, 0.0275);

    // 配对条件
    nh.param("max_y_diff", current_params.max_y_diff, 0.25);
    nh.param("max_angle_diff", current_params.max_angle_diff, 6.0);
    nh.param("max_bar_length_diff", current_params.max_bar_length_diff, 0.20);
    nh.param("max_bar_width_diff", current_params.max_bar_width_diff, 0.22);
    nh.param("min_v_control", current_params.min_v_control, 0.93);
    nh.param("max_armor_angle", current_params.max_armor_angle, 6.0);
    nh.param("min_armor_angle", current_params.min_armor_angle, 1.0);

    // 手动触发一次回调，把 YAML 的值喂给算法
    reconfigCallback(current_params, 0);

    // ================= [新增] 启动动态调参 Server =================
    dynamic_reconfigure::Server<rm_vision::ArmorParamsConfig> server;
    dynamic_reconfigure::Server<rm_vision::ArmorParamsConfig>::CallbackType f;
    f = boost::bind(&reconfigCallback, _1, _2);
    server.setCallback(f);

    //用image_transport订阅
    image_transport::ImageTransport it(nh);
    
    //数据处理和相机内参
    image_transport::Subscriber image_sub = it.subscribe(image_topic,5,imageCallback,
        image_transport::TransportHints("compressed"));
    ros::Subscriber info_sub = nh.subscribe(camera_info_topic,10,cameraInfoCallback);
    //用全局公开话题名，方便其他工具订阅
    debug_image_pub = nh.advertise<sensor_msgs::Image>("/ft/debug/image",1);
    //发布消息
    detection_pub = nh.advertise<ft_detection::DetectionFrame>("/ft/detection",10);
    // 启动时打印实际订阅的话题，方便确认参数是否生效
    ROS_INFO_ONCE("Image subscribe started. Image: %s, CameraInfo: %s",
        image_topic.c_str(),camera_info_topic.c_str());
    
    ros::spin();

    return 0;
}
//内参回调
void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
    last_camera_K = cv::Mat(3, 3, CV_64F);
    std::copy(msg->K.begin(), msg->K.end(), last_camera_K.ptr<double>());
    
    const int d_size = static_cast<int>(msg->D.size());
    last_dist_D = cv::Mat(1, std::max(d_size, 1), CV_64F, cv::Scalar(0));//防止相机发出D的长度是0
    std::copy(msg->D.begin(), msg->D.end(), last_dist_D.ptr<double>());

    last_image_width  = static_cast<int>(msg->width);
    last_image_height = static_cast<int>(msg->height);
    camera_info_received = true;

    solver.setCameraInfo(last_camera_K,last_dist_D,
                        cv::Size(last_image_width,last_image_height));
    
}
//图像回调
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{//解码 → 打印 → 发布调试图像
    //保存时间戳
    last_image_stamp = msg->header.stamp;
    last_frame_id = msg->header.frame_id;
    //用toshare实现零拷贝
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvShare(msg,"bgr8");
    }catch(cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s",e.what());
        return;
    }
    //用于测试
    ROS_INFO_ONCE("first frame: encoding=%s size=%ux%u", msg->encoding.c_str(), msg->width, msg->height);
    
    //分析阶段只读：src 是消息缓冲区的只读视图，detector.detect 的形参是 const 引用
    const cv::Mat& src = cv_ptr->image;
    const std::vector<Armor> armors = detector.detect(src);

    //先把图像拷进待发布消息的缓冲区，然后直接在这个缓冲区上画框 —— 整个回调只发生一次拷贝。
    sensor_msgs::ImagePtr out_image(new sensor_msgs::Image);
    out_image->header       = msg->header;
    out_image->height       = static_cast<uint32_t>(src.rows);
    out_image->width        = static_cast<uint32_t>(src.cols);
    out_image->encoding     = "bgr8";
    out_image->is_bigendian = 0;//小端字节序（普通电脑默认的）
    out_image->step         = static_cast<uint32_t>(src.cols * src.elemSize());
    out_image->data.resize(out_image->step * src.rows);
    //step是一行图像数据占用的总字节数
    //src.elemSize()：每个像素占用多少个字节（比如 bgr8 格式，3通道，每个通道 8位即 1字节，所以 elemSize() 是 3）

    // 用一个 cv::Mat 包住消息的 data（不拷贝），再 copyTo 进去
    cv::Mat canvas(src.rows, src.cols, src.type(), out_image->data.data(), out_image->step);
    src.copyTo(canvas);                    // 唯一的一次拷贝
    detector.drawArmors(canvas, armors);   // 画框 + 序号 + 大小

    //发布检测消息
    ft_detection::DetectionFrame frame_msg;//创建消息内容
    frame_msg.header.stamp = msg->header.stamp;
    frame_msg.header.frame_id = last_frame_id;

    //求解位姿 + 画轴 + 广播 TF + 发布检测消息
    int tf_id = 0;
    for(const auto& armor : armors){
        ArmorSolver::Pose pose;
        if(!solver.solve(armor,src.size(),pose)) continue;//坏解直接跳过
        solver.drawAxes(canvas,pose);// 画在 canvas 上，会随 /ft/debug/image 发出去
        solver.broadcastTF(pose,msg->header,"armor"+std::to_string(tf_id));

        //拦截Nan
        if(std::isnan(pose.tvec[0]) || std::isnan(pose.tvec[1]) || std::isnan(pose.tvec[2]) ||
           std::isnan(pose.q.x()) || std::isnan(pose.q.y()) || std::isnan(pose.q.z()) ||
           std::isnan(pose.q.w())){
            continue;
        }
        ft_detection::Detection detection_msg;
        detection_msg.id = tf_id;
        detection_msg.is_red = (armor.color == "Red");
        detection_msg.is_large_armor = (armor.type == "Big");
        //位置
        detection_msg.pose.position.x = pose.tvec[0];
        detection_msg.pose.position.y = pose.tvec[1];
        detection_msg.pose.position.z = pose.tvec[2];
        //四元数
        detection_msg.pose.orientation.x = pose.q.x();
        detection_msg.pose.orientation.y = pose.q.y();
        detection_msg.pose.orientation.z = pose.q.z();
        detection_msg.pose.orientation.w = pose.q.w();
        //存入数据
        for (int i = 0; i < 4; ++i) {
            detection_msg.corners[i].x = armor.corners[i].x;
            detection_msg.corners[i].y = armor.corners[i].y;
            detection_msg.corners[i].z = 0.0;
        }
        detection_msg.confidence = 1.0;
        frame_msg.detections.push_back(detection_msg);

        //顶点、和 ID
        std::string id_pub = "ID:" + std::to_string(detection_msg.id);
        int baseline = 0;//防止字超出屏幕
        cv::Size text_size = cv::getTextSize(id_pub,cv::FONT_HERSHEY_SIMPLEX,
                                             0.51,2,&baseline);
        int text_y = armor.bounding_box.y - 20;
        if(text_y - text_size.height < 0){
            text_y = armor.bounding_box.y + armor.bounding_box.height + text_size.height + 5;
        }

        cv::putText(canvas,id_pub,cv::Point(armor.bounding_box.x,text_y),
                    cv::FONT_HERSHEY_SIMPLEX,0.51,cv::Scalar(0,255,0),2);
        for(const auto& c : armor.corners){
            cv::circle(canvas,c,3,cv::Scalar(0,255,0),-1);
        }
        tf_id += 1;
    }
    
    ROS_INFO_THROTTLE(1.0, "Armors found: %zu | camera info: %s",
        armors.size(), camera_info_received ? "YES" : "NO");

    //发布调试图像（消息里已经是画好框的图，publish 不再拷贝）
    debug_image_pub.publish(out_image);
    //发布检测信息
    detection_pub.publish(frame_msg);
}

void reconfigCallback(rm_vision::ArmorParamsConfig &config,uint32_t level)
{
    current_params = config;
    //更新 armor_detection 参数
    ArmorDetector::Params det_params;
    det_params.red_h_min1 = config.red_h_min1;
    det_params.red_h_max1 = config.red_h_max1;
    det_params.red_h_min2 = config.red_h_min2;
    det_params.red_h_max2 = config.red_h_max2;
    det_params.blue_h_min = config.blue_h_min;
    det_params.blue_h_max = config.blue_h_max;
    det_params.s_min = config.s_min;
    det_params.s_max = config.s_max;
    det_params.v_min = config.v_min;
    det_params.v_max = config.v_max;
    det_params.min_area = config.min_area;
    det_params.min_bar_ratio = config.min_bar_ratio;
    det_params.max_bar_ratio = config.max_bar_ratio;
    det_params.max_y_diff = config.max_y_diff;
    det_params.max_angle_diff = config.max_angle_diff;
    det_params.max_bar_length_diff = config.max_bar_length_diff;
    det_params.max_bar_width_diff = config.max_bar_width_diff;
    det_params.min_v_control = config.min_v_control;
    det_params.max_armor_angle = config.max_armor_angle;
    det_params.min_armor_angle = config.min_armor_angle;
    detector.setParams(det_params);
    //更新 armor_solver 参数
    ArmorSolver::Params solver_params;
    solver_params.small_half_w = config.small_half_w;
    solver_params.big_half_w = config.big_half_w;
    solver_params.bar_half_len = config.bar_half_len;
    solver.setParams(solver_params);
}