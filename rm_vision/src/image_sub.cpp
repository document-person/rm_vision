#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "rm_vision/armor_detector.h"
#include "rm_vision/armor_solver.h"


//发布解码后的图像
ros::Publisher debug_image_pub;
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

int main(int argc,char* argv[])
{
    ros::init(argc,argv,"image_sub");//节点名字为image_sub
    ros::NodeHandle nh("~");//用私有句柄

        // TF 父坐标系：默认 camera_optical_frame（机器人 TF 树上有这一帧）
    std::string tf_parent;
    nh.param<std::string>("tf_parent_frame", tf_parent, "camera_optical_frame");                                            
    solver.setTfParent(tf_parent);
    ROS_INFO_THROTTLE(2.0,"TF parent frame = [%s]", tf_parent.c_str());
    
    //从参数服务器读取话题名（预留接口，随时可改）
    std::string image_topic, camera_info_topic;
    nh.param<std::string>("image_topic",image_topic,"/hk_camera/image_raw");
    nh.param<std::string>("camera_info_topic",camera_info_topic,"/hk_camera/camera_info");
    //用image_transport订阅
    image_transport::ImageTransport it(nh);
    
    //数据处理和相机内参
    image_transport::Subscriber image_sub = it.subscribe(image_topic,5,imageCallback,
        image_transport::TransportHints("compressed"));
    ros::Subscriber info_sub = nh.subscribe(camera_info_topic,10,cameraInfoCallback);
    //用全局公开话题名，方便其他工具订阅
    debug_image_pub = nh.advertise<sensor_msgs::Image>("/ft/debug/image",1);
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

    //求解位姿 + 画轴 + 广播 TF
    int tf_id = 0;
    for(const auto& armor : armors){
        ArmorSolver::Pose pose;
        if(!solver.solve(armor,src.size(),pose)) continue;//坏解直接跳过
        solver.drawAxes(canvas,pose);// 画在 canvas 上，会随 /ft/debug/image 发出去
        solver.broadcastTF(pose,msg->header,"armor"+std::to_string(tf_id));
        tf_id += 1;
    }

    ROS_INFO_THROTTLE(1.0, "Armors found: %zu | camera info: %s",
        armors.size(), camera_info_received ? "YES" : "NO");

    //发布调试图像（消息里已经是画好框的图，publish 不再拷贝）
    debug_image_pub.publish(out_image);
}