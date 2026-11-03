#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

//发布解码后的图像
ros::Publisher debug_image_pub;
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
int last_ima_width = 0;
int last_ima_height = 0;

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
void imageCallback(const sensor_msgs::ImageConstPtr& msg);

int main(int argc,char* argv[])
{
    ros::init(argc,argv,"image_sub");//节点名字为image_sub
    ros::NodeHandle nh("~");//用私有句柄
    
    //从参数服务器读取话题名（预留接口，随时可改）
    std::string image_topic, camera_info_topic;
    nh.param<std::string>("image_topic",image_topic,"/hk_camera/image_raw");
    nh.param<std::string>("camera_info_topic",camera_info_topic,"/hk_camera/camera_info");
    //用image_transport订阅
    image_transport::ImageTransport it(nh);
    
    //压缩图像和相机内参
    image_transport::Subscriber image_sub = it.subscribe(image_topic,5,imageCallback,
        image_transport::TransportHints("compressed"));
    ros::Subscriber info_sub = nh.subscribe(camera_info_topic,10,cameraInfoCallback);
    //用全局公开话题名，方便其他工具订阅
    debug_image_pub = nh.advertise<sensor_msgs::Image>("/ft/debug/image",1);
    // 启动时打印实际订阅的话题，方便确认参数是否生效
    ROS_INFO("Image subscribe started. Image: %s, CameraInfo: %s",
        image_topic.c_str(),camera_info_topic.c_str());
    
    ros::spin();

    return 0;
}
//内参回调
void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{//提取参数
    last_camera_K = cv::Mat(3,3,CV_64F,(void*)msg->K.data()).clone();
    last_dist_D = cv::Mat(1,msg->D.size(),CV_64F,(void*)msg->D.data()).clone();
    last_ima_width = msg->width;
    last_ima_height = msg->height;
    camera_info_received = true;
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
    ROS_INFO("Image: %dx%d | stamp: %.3f | frame_id: %s | info: %s",
        cv_ptr->image.cols,cv_ptr->image.rows,last_image_stamp.toSec(),
        last_frame_id.c_str(),camera_info_received ? "YES" :"NO");
    //发布调试图像
    debug_image_pub.publish(cv_ptr->toImageMsg());
}