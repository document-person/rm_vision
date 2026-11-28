#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "ft_detector/armor_detector.h"
#include "ft_detector/armor_solver.h"
#include <ft_detection/DetectionFrame.h>
#include <ft_detector/ArmorParamsConfig.h>
#include <dynamic_reconfigure/server.h>
#include "ft_detector/armor_detector_nn.h"


ros::Publisher debug_image_pub;
ros::Publisher detection_pub;
ArmorDetector detector;
ArmorDetectorNN nn_detector;
std::string detector_mode = "traditional";
std::string compare_mode;
ArmorSolver solver;

std::string last_frame_id;
bool camera_info_received = false;

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
void imageCallback(const sensor_msgs::ImageConstPtr& msg);
void reconfigCallback(ft_detector::ArmorParamsConfig &config,uint32_t level);

int main(int argc,char* argv[])
{
    ros::init(argc,argv,"image_sub");
    ros::NodeHandle nh("~");

    std::string image_topic, camera_info_topic;
    std::string tf_parent;

    nh.param<std::string>("image_topic",image_topic,"/hk_camera/image_raw");
    nh.param<std::string>("camera_info_topic",camera_info_topic,"/hk_camera/camera_info");
    nh.param<std::string>("tf_parent_frame", tf_parent, "camera_optical_frame");                                            
    solver.setTfParent(tf_parent);

    nh.param<std::string>("detector_mode", detector_mode, "traditional");
    nh.param<std::string>("compare_mode", compare_mode, "");
    if (detector_mode != "traditional" && detector_mode != "nn") {
        ROS_WARN("unknown detector_mode '%s', fallback", detector_mode.c_str());
        detector_mode = "traditional";
    }
    if (detector_mode == "nn" && !nn_detector.initFromRos(nh)) {
        ROS_ERROR("NN init failed, fallback to traditional");
        detector_mode = "traditional";
    }
    ROS_INFO("detector_mode=%s", detector_mode.c_str());
    
    // 动态参数：默认值只来自 cfg（ArmorParamsConfig 构造时已按 cfg 填好）。
    // 用 getParam：YAML 里没有的键不会改动 cfg 的默认值，也不会把默认值再抄一份到代码里。
    ft_detector::ArmorParamsConfig cfg;

    // 颜色阈值(HSV)
    nh.getParam("red_h_min1", cfg.red_h_min1);
    nh.getParam("red_h_max1", cfg.red_h_max1);
    nh.getParam("red_h_min2", cfg.red_h_min2);
    nh.getParam("red_h_max2", cfg.red_h_max2);
    nh.getParam("blue_h_min", cfg.blue_h_min);
    nh.getParam("blue_h_max", cfg.blue_h_max);
    nh.getParam("s_min", cfg.s_min);
    nh.getParam("s_max", cfg.s_max);
    nh.getParam("v_min", cfg.v_min);
    nh.getParam("v_max", cfg.v_max);

    // 灯条过滤
    nh.getParam("min_area", cfg.min_area);
    nh.getParam("min_bar_ratio", cfg.min_bar_ratio);
    nh.getParam("max_bar_ratio", cfg.max_bar_ratio);
    nh.getParam("max_bar_angle", cfg.max_bar_angle);
    nh.getParam("min_fill_ratio", cfg.min_fill_ratio);

    // 装甲板物理尺寸
    nh.getParam("small_half_w", cfg.small_half_w);
    nh.getParam("big_half_w", cfg.big_half_w);
    nh.getParam("bar_half_len", cfg.bar_half_len);

    // 配对条件
    nh.getParam("max_y_diff", cfg.max_y_diff);
    nh.getParam("max_angle_diff", cfg.max_angle_diff);
    nh.getParam("max_bar_length_diff", cfg.max_bar_length_diff);
    nh.getParam("max_bar_width_diff", cfg.max_bar_width_diff);
    nh.getParam("min_v_control", cfg.min_v_control);
    nh.getParam("max_x_dist_ratio", cfg.max_x_dist_ratio);
    nh.getParam("min_x_dist_ratio", cfg.min_x_dist_ratio);
    nh.getParam("bar_size_ratio", cfg.bar_size_ratio);

    //解算器参数
    nh.getParam("max_reproj_error", cfg.max_reproj_error);
    nh.getParam("force_vertical_plate", cfg.force_vertical_plate);
    nh.getParam("memory_timeout", cfg.memory_timeout);

    //先建服务器：setCallback 会立即用服务器当前配置回调一次，再把 YAML 值 update 下去。
    //不要在 setCallback 前手工调用 reconfigCallback —— 那一次会被这两步各覆盖一遍。
    dynamic_reconfigure::Server<ft_detector::ArmorParamsConfig> server;
    dynamic_reconfigure::Server<ft_detector::ArmorParamsConfig>::CallbackType f;
    f = boost::bind(&reconfigCallback, _1, _2);
    server.setCallback(f);
    server.updateConfig(cfg);

    //用image_transport订阅
    image_transport::ImageTransport it(nh);
    
    //数据处理和相机内参
    int queue_size;
    nh.param("image_queue_size", queue_size, 1);
    // 默认走 compressed，但允许标准参数 ~image_transport 覆盖（如 roslaunch ... image_transport:=raw）
    image_transport::TransportHints hints("compressed", ros::TransportHints(), nh);
    image_transport::Subscriber image_sub = it.subscribe(image_topic,queue_size,
                imageCallback,hints);
    // 内参与图像同帧率发布，队列用同一个参数即可
    ros::Subscriber info_sub = nh.subscribe(camera_info_topic,queue_size,cameraInfoCallback);
    debug_image_pub = nh.advertise<sensor_msgs::Image>("/ft/debug/image",1);
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
    cv::Mat K(3, 3, CV_64F);
    std::copy(msg->K.begin(), msg->K.end(), K.ptr<double>());

    const int d_size = static_cast<int>(msg->D.size());
    cv::Mat D(1, std::max(d_size, 1), CV_64F, cv::Scalar(0));//防止相机发出D的长度是0
    std::copy(msg->D.begin(), msg->D.end(), D.ptr<double>());
    
    camera_info_received = true;
    solver.setCameraInfo(K, D, cv::Size(static_cast<int>(msg->width),
                                       static_cast<int>(msg->height)));
}
//图像回调
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    //只保存 frame_id，后面填 DetectionFrame 和 TF 用（时间戳直接用 msg->header）
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
    
    //分析阶段：src 是消息缓冲区的只读视图，detector.detect 的形参是 const 引用
    const cv::Mat& src = cv_ptr->image;
    std::vector<Armor> armors;
    if(detector_mode == "nn"){
        armors = nn_detector.detect(src);
    }else armors = detector.detect(src);

    //先把图像拷进待发布消息的缓冲区，然后直接在这个缓冲区上画框 —— 整个回调只发生一次拷贝。
    sensor_msgs::ImagePtr out_image(new sensor_msgs::Image);
    out_image->header       = msg->header;
    out_image->height       = static_cast<uint32_t>(src.rows);
    out_image->width        = static_cast<uint32_t>(src.cols);
    out_image->encoding     = "bgr8";
    out_image->is_bigendian = 0;//小端字节序（默认）
    out_image->step         = static_cast<uint32_t>(src.cols * src.elemSize());
    out_image->data.resize(out_image->step * src.rows);
    //step是一行图像数据占用的总字节数
    //src.elemSize()：每个像素占用多少个字节（比如 bgr8 格式，3通道，每个通道 8位即 1字节，所以 elemSize() 是 3）

    // 用一个 cv::Mat 包住消息的 data（不拷贝），再 copyTo 进去
    cv::Mat canvas(src.rows, src.cols, src.type(), out_image->data.data(), out_image->step);
    src.copyTo(canvas);                    // 唯一的一次拷贝
    if(detector_mode == "nn"){
        nn_detector.drawArmors(canvas,armors);
    }else detector.drawArmors(canvas, armors);

    
    if (!compare_mode.empty() && compare_mode != detector_mode) {
        const auto t_cmp = std::chrono::steady_clock::now();
        const std::vector<Armor> other = (compare_mode == "nn") ? nn_detector.detect(src)
                                                                : detector.detect(src);
        const double cmp_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_cmp).count();
        const cv::Scalar col = (compare_mode == "nn") ? cv::Scalar(255, 0, 255)
                                                      : cv::Scalar(255, 255, 0);
        for (const auto& a : other) {
            std::vector<cv::Point> poly;
            for (const auto& c : a.corners) poly.emplace_back(cvRound(c.x), cvRound(c.y));
            cv::polylines(canvas, poly, true, col, 2);
        }
        cv::putText(canvas, cv::format("%s: %zu (%.2f ms)", compare_mode.c_str(),
                                       other.size(), cmp_ms),
                    cv::Point(10, canvas.rows - 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
    }


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
        // 传统检测 confidence 固定 1.0；NN 填的是真实颜色分数
        detection_msg.confidence = armor.confidence;
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

void reconfigCallback(ft_detector::ArmorParamsConfig &config,uint32_t level)
{
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
    //灯条
    det_params.min_area = config.min_area;
    det_params.min_bar_ratio = config.min_bar_ratio;
    det_params.max_bar_ratio = config.max_bar_ratio;
    det_params.max_bar_angle = config.max_bar_angle;
    det_params.min_fill_ratio = config.min_fill_ratio;
    //装甲板
    det_params.max_y_diff = config.max_y_diff;
    det_params.max_angle_diff = config.max_angle_diff;
    det_params.max_bar_length_diff = config.max_bar_length_diff;
    det_params.max_bar_width_diff = config.max_bar_width_diff;
    det_params.min_v_control = config.min_v_control;
    det_params.max_x_dist_ratio = config.max_x_dist_ratio;
    det_params.min_x_dist_ratio = config.min_x_dist_ratio;
    det_params.bar_size_ratio = config.bar_size_ratio;
    detector.setParams(det_params);
    //更新 armor_solver 参数
    ArmorSolver::Params solver_params;
    solver_params.small_half_w = config.small_half_w;
    solver_params.big_half_w = config.big_half_w;
    solver_params.bar_half_len = config.bar_half_len;
    solver_params.max_reproj_error = config.max_reproj_error;
    solver_params.force_vertical_plate = config.force_vertical_plate;
    solver_params.memory_timeout = config.memory_timeout;
    solver.setParams(solver_params);
}