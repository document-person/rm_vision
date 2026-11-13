#include <ros/ros.h>
#include <cmath>
#include <algorithm>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include "rm_vision/armor_solver.h"

ArmorSolver::ArmorSolver(){}
//提取相机的参数
void ArmorSolver::setCameraInfo(const cv::Mat& K,const cv::Mat& D,
                                const cv::Size& image_size)
{
    if(K.rows != 3 || K.cols != 3){//防御
        ready_ = false;
        return;
    }
    K.convertTo(K_,CV_64F);// 统一成 double，后面的计算要用double类型
    if(D.empty()){//手搓一个
        D_ = cv::Mat::zeros(1,5,CV_64F);
    }else D.convertTo(D_,CV_64F);
    image_size_ = image_size;
    ready_ = (K_.at<double>(0,0) > 1.0 && K_.at<double>(1,1) > 1.0);
}
//确定灯条四个点的3D坐标
std::vector<cv::Point3f> ArmorSolver::find3DPoints(const std::string& type)
{
    const float half_w = (type == "Big") ? params_.big_half_w : params_.small_half_w;
    const float half_h = params_.bar_half_len;
    return{
        {-half_w,-half_h,0.0f},//左上
        {half_w,-half_h,0.0f}, //右上
        {half_w,half_h,0.0f},  //右下
        {-half_w,half_h,0.0f}, //左下
    };
}
//检查角点顺序
bool ArmorSolver::cornerscheck(const std::vector<cv::Point2f>& corners)
{
    if(corners.size() != 4) return false;
    return (corners[0].x < corners[1].x) && (corners[0].y < corners[3].y);
}
//重投影误差计算
double ArmorSolver::reprojectionRms(const std::vector<cv::Point3f>& obj,
                                    const std::vector<cv::Point2f>& corner,
                                    const cv::Vec3d& rvec, const cv::Vec3d& tvec) const
{
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj,rvec,tvec,K_,D_,proj);//重投影函数
    double sq = 0.0;
    for(size_t i = 0;i < proj.size() && i < corner.size();++i){
        const cv::Point2f d = proj[i] - corner[i];
        sq += d.x * d.x + d.y * d.y;//方差计算
    }
    return std::sqrt(sq / static_cast<double>(proj.size()));
}

//开始解算3D位姿和发布TF
bool ArmorSolver::solve(const Armor& armor,const cv::Size& image_size,Pose& out)
{
    //超过0.5秒没有解算就忘记上一帧的记忆，用于后面的防止大小板误判
    if(has_last_ && (ros::Time::now() - last_time_).toSec() > params_.memory_timeout){
        has_last_ = false;
    }
    out = Pose{};//清空上一轮的输出，防止数据残留
    if (!ready_){
        ROS_WARN_THROTTLE(2.0,"ArmorSolver: Don't get the camera info");
        return false;
    }
    //
    if(image_size != image_size_){
        ROS_WARN_THROTTLE(2.0,"ArmorSolver: image's %dx%d don't equal to camera_info's %dx%d",
            image_size.width,image_size.height,
            image_size_.width,image_size_.height);
        return false;
    }
    if(armor.corners.size() != 4) return false;
    if(armor.type != "Big" && armor.type != "Small") return false;
    //检查点的顺序，但是只警告，防止因为极端斜视过滤正确的板
    if(!cornerscheck(armor.corners)){
        ROS_WARN_THROTTLE(1.0,"ArmorSolver: Corner order might be mirrored (tl/tr/bl relationship is invalid)");
    }
    const std::vector<cv::Point3f> obj = find3DPoints(armor.type);
    //运算PnP
    std::vector<cv::Vec3d> rvecs,tvecs;
    const int n = cv::solvePnPGeneric(obj,armor.corners,K_,D_,
                                      rvecs,tvecs,false,cv::SOLVEPNP_IPPE);
                                //SOLVEPNP_IPPE 是专门针对平面物体的无初值解法，极其稳定
    if(n <= 0){
        ROS_WARN_THROTTLE(1.0,"ArmorSolver: solvePnp failed");
        return false;
    }
    //选解
    struct Cand{
        int idx;// 原始解数组的索引
        double err;// 重投影误差
        cv::Vec3d nz;// 板子法向量
    };
    std::vector<Cand> cands;
    for(int k = 0;k < n;++k){
        if(tvecs[k][2] < 0.1) continue;
        cv::Mat R;
        cv::Rodrigues(rvecs[k],R);
        const cv::Vec3d nz(R.at<double>(0,2),R.at<double>(1,2),R.at<double>(2,2));
        if(nz.dot(tvecs[k]) < 0.0) continue;
        const double err = reprojectionRms(obj,armor.corners,rvecs[k],tvecs[k]);
        cands.push_back({k,err,nz});
    }
    if(cands.empty()){
        ROS_WARN_THROTTLE(1.0,"ArmorSolver: No valid solution");
        return false;
    }
    std::sort(cands.begin(),cands.end(),
              [](const Cand& a,const Cand& b){return a.err < b.err;});
    
              
    //调试：打印两个候选解的差异
    // for(size_t i = 0; i < cands.size(); ++i){
    // ROS_INFO_STREAM_THROTTLE(1.0,
    //     "  cand[" << i << "] err=" << cands[i].err
    //     << " z=" << tvecs[cands[i].idx][2]
    //        << " nz=(" << cands[i].nz[0] << "," << cands[i].nz[1] << "," << cands[i].nz[2]<< ")");
    // }
    
    //两个解误差接近时（平面镜像歧义），用上一帧的法向裁决
    int pick = 0;
    if(cands.size() >= 2 && has_last_){
        const double r = cands[1].err / std::max(cands[0].err ,1e-6);//防止分母是0
        if(r < 1.5 && cands[0].nz.dot(last_nz_) < cands[1].nz.dot(last_nz_)){
            pick = 1;// 选和上一帧朝向更一致的那个
        }
    }
    out.rvec = rvecs[cands[pick].idx];
    out.tvec = tvecs[cands[pick].idx];
    //LM精修
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,20,1e-6);
    cv::solvePnPRefineLM(obj,armor.corners,K_,D_,out.rvec,out.tvec,criteria);
    //算一下精修完的误差
    out.reproj_rms = reprojectionRms(obj,armor.corners,out.rvec,out.tvec);
    out.valid = (out.reproj_rms < params_.max_reproj_error);
    //把旋转向量变成旋转矩阵
    cv::Mat R;
    cv::Rodrigues(out.rvec,R);
    for(int r = 0;r < 3;++r){
        for(int c = 0;c < 3;++c){
            out.R(r,c) = R.at<double>(r,c);//把堆的数据放到栈上面
        }
    }
    // 打开这个开关 = 用物理先验把俯仰/横滚按死，只保留偏航，再用"高度定 z、板心定 x,y"重算平移
    if(params_.force_vertical_plate){
        const double h_px_v = cv::norm(armor.corners[3] - armor.corners[0]);
        if(h_px_v > 1.0){//取值1.0,防止因为数据太小而不可靠
            const double yaw_v = std::atan2(out.R(0,2), out.R(2,2));  //只取偏航角
            const double cv_ = std::cos(yaw_v), sv_ = std::sin(yaw_v);
            out.R = cv::Matx33d( cv_, 0.0, sv_,
                                 0.0, 1.0, 0.0,
                                 -sv_, 0.0, cv_);
            cv::Rodrigues(out.R, out.rvec);
            out.tvec[2] = K_.at<double>(1,1) * (2.0*obj[2].y) / h_px_v;
            const cv::Point2f center = (armor.corners[0] + armor.corners[1] +
                                     armor.corners[2] + armor.corners[3]) * 0.25f;
            //反投影公式计算x和y
            out.tvec[0]=(center.x - K_.at<double>(0,2)) * out.tvec[2] / K_.at<double>(0,0);
            out.tvec[1]=(center.y - K_.at<double>(1,2)) * out.tvec[2] / K_.at<double>(1,1);
        }
    }


    //调试：打印姿态角
    //     const double yaw   = std::atan2(out.R(1,0), out.R(0,0)) * 180.0 / CV_PI;
    // const double pitch = std::asin(std::max(-1.0, std::min(1.0, -out.R(2,0)))) * 180.0 / CV_PI;
    // const double roll  = std::atan2(out.R(2,1), out.R(2,2)) * 180.0 / CV_PI;
    // ROS_INFO_STREAM_THROTTLE(1.0,
    //     "  yaw=" << yaw << "  pitch=" << pitch << "  roll=" << roll
    //     << "  nz=(" << out.R(0,2) << "," << out.R(1,2) << "," << out.R(2,2) << ")");
//我们的约定下，板子正对相机时必须是 yaw≈0, pitch≈0, roll≈0。这一条是硬判据

    //更新帧间记忆
    last_nz_ = cv::Vec3d(out.R(0,2),out.R(1,2),out.R(2,2));
    last_time_ = ros::Time::now();
    has_last_ = true;
    
    //旋转矩阵转换为四元数
    // tf2::Matrix3x3 的构造是「行主序」9 个数：xx,xy,xz, yx,yy,yz, zx,zy,zz
    const tf2::Matrix3x3 m(out.R(0,0),out.R(0,1),out.R(0,2),
                           out.R(1,0),out.R(1,1),out.R(1,2),
                           out.R(2,0),out.R(2,1),out.R(2,2));
    m.getRotation(out.q);//转换函数
    out.q.normalize();//数值误差会让模长偏离1，rviz 会报警，所以用这个函数把模长变成1

    // // 调试使用：像素宽高比、偏航角、两个方向分别反推的距离
    // const double w_px   = cv::norm(armor.corners[1] - armor.corners[0]);
    // const double h_px   = cv::norm(armor.corners[3] - armor.corners[0]);
    // const double half_w = obj[1].x;    // 物体点半宽（米）
    // const double half_h = obj[2].y;    // 物体点半高（米）
    // // 分别用"宽"和"高"反推距离：板子正对时两者必须相等，并且约等于 tvec.z。
    // // 两者差得越多，说明物体点的宽高比和角点的宽高比越不匹配。
    // const double z_from_w = K_.at<double>(0,0) * (2.0*half_w) / w_px;
    // const double z_from_h = K_.at<double>(1,1) * (2.0*half_h) / h_px;
    // // 板法向 = R 的第三列。正对时 nz=(0,0,1)，yaw/pitch 都应该是 0。
    // const double yaw   = std::atan2(out.R(0,2), out.R(2,2)) * 180.0 / CV_PI;
    // const double pitch = std::asin(std::max(-1.0, std::min(1.0, -out.R(1,2)))) * 180.0 / CV_PI;
    // // 整段数据里 w/h 的最大值对应"板子最正对"的那一帧，此时 w/h = W/L，
    // // 用它反推灯条长度 L：L = W / max(w/h)。
    // const double aspect = w_px / std::max(h_px, 1e-6);
    // if(aspect > max_aspect_) max_aspect_ = aspect;
    // ROS_INFO_STREAM_THROTTLE(1.0,
    //     "pose: " << armor.type << "  sols=" << n
    //     << "  t=(" << out.tvec[0] << ", " << out.tvec[1] << ", " << out.tvec[2] << ") m"
    //     << "  rms=" << out.reproj_rms << " px");
    // ROS_INFO_STREAM_THROTTLE(1.0,
    //     "  w=" << w_px << " h=" << h_px << " w/h=" << aspect
    //     << "  zW=" << z_from_w << " zH=" << z_from_h
    //     << "  yaw=" << yaw << " pitch=" << pitch);
    // ROS_INFO_STREAM_THROTTLE(5.0,
    //     "  [calib] max w/h=" << max_aspect_
    //     << "  => L=" << (2.0*half_w*1000.0)/max_aspect_ << " mm"
    //     << "  (当前设定 " << (2.0*half_h*1000.0) << " mm)");
    
    return out.valid;
}

//画坐标轴
void ArmorSolver::drawAxes(cv::Mat& image,const Pose& pose) const
{
    if(!pose.valid) return;
    const float L = 0.05f;//轴长 5cm，约板宽的 1/5
    const std::vector<cv::Point3f> axis = {
        {0.0f,0.0f,0.0f},//原点
        {L   ,0.0f,0.0f},//+X 沿板宽向右
        {0.0f, L  ,0.0f}, //+Y 沿板高向下
        {0.0f,0.0f, L}//+Z 垂直于板面指向板里
    };
    std::vector<cv::Point2f> proj;
    cv::projectPoints(axis,pose.rvec,pose.tvec,K_,D_,proj);
    const cv::Point2f& o = proj[0];
    cv::line(image,o,proj[1],cv::Scalar(0,0,225),2);//X 红
    cv::line(image,o,proj[2],cv::Scalar(0,225,0),2);//Y 绿
    cv::line(image,o,proj[3],cv::Scalar(225,0,0),2);//Z 蓝
    cv::circle(image,o,3,cv::Scalar(0,225,225),-1);//原点 黄
    cv::putText(image,cv::format("%.2fm",pose.tvec[2]),
                o + cv::Point2f(6,-8),cv::FONT_HERSHEY_SIMPLEX,0.45,
                cv::Scalar(0,255,255),1);
    
    // // 调试：放大 1.5 倍的物体矩形，应和板子同心且平行
    // const float s = 1.5f, hw = 0.1175f, hh = 0.0275f;
    // const std::vector<cv::Point3f> box = {
    //     {-hw*s,-hh*s,0},{hw*s,-hh*s,0},{hw*s,hh*s,0},{-hw*s,hh*s,0}
    // };
    // std::vector<cv::Point2f> bp;
    // cv::projectPoints(box, pose.rvec, pose.tvec, K_, D_, bp);
    // for(int i = 0; i < 4; ++i)
    //     cv::line(image, bp[i], bp[(i+1)%4], cv::Scalar(0,255,255), 1);
}

void ArmorSolver::broadcastTF(const Pose& pose,const std_msgs::Header& header,
                              const std::string& child_frame)
{
    if(!pose.valid) return;
    // 父坐标系优先用参数指定的；没指定才退回图像自带的 frame_id
    std_msgs::Header tf_header = header;
    if(!tf_parent_.empty()) tf_header.frame_id = tf_parent_;
    if(header.frame_id.empty()){
        ROS_WARN_THROTTLE(2.0,"TF:image header.frame_id is empty,check the camera driver");
        return;
    }
    if(!tf_broadcaster_){//懒加载：如果广播器没创建，创建一个
        tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
    }
    geometry_msgs::TransformStamped ts;
    ts.header = tf_header;//时间戳+父坐标系
    ts.child_frame_id = child_frame;
    ts.transform.translation.x = pose.tvec[0];
    ts.transform.translation.y = pose.tvec[1];
    ts.transform.translation.z = pose.tvec[2];
    ts.transform.rotation.x = pose.q.x();
    ts.transform.rotation.y = pose.q.y();
    ts.transform.rotation.z = pose.q.z();
    ts.transform.rotation.w = pose.q.w();

    tf_broadcaster_->sendTransform(ts);
}