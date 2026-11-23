#include <ros/ros.h>
#include <ft_detection/DetectionFrame.h>
#include <ft_tracker/TrackArray.h>
#include <opencv2/core.hpp>
#include <ft_tracker/tracker.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ft_tracker/ArmorParamsConfig.h>
#include <dynamic_reconfigure/server.h>
#include <boost/bind.hpp>

#include <cmath>
#include <vector>
#include <string>

ros::Publisher track_pub;
ft_tracker::Tracker tracker;
ft_tracker::Tracker::Params params;
ros::Publisher marker_pub;
std::string g_tf_frame = "camera_optical_frame";//统一输出坐标系
ft_tracker::ArmorParamsConfig current_params;
double g_reacq_time = 1.5;//断档后最多外推几秒
double g_reacq_dist = 0.8;//允许接回的最大距离偏差
double g_id_memory  = 4.0;//断档后还能保留 id 的最长时间

//检验位置是否合理(有过预测飞到外面的经历)
static bool plausiblePos(const double p[3])
{
    const double z = p[2];
    if(z < 0.5 || z > 15.0) return false;
    if(std::fabs(p[0]) > 0.6 * z) return false;
    if(std::fabs(p[1]) > 0.5 * z) return false;
    return true;
}

void trackCallback(const ft_detection::DetectionFrame::ConstPtr& msg);
void reconfigCallback(ft_tracker::ArmorParamsConfig &config, uint32_t level);

int main(int argc, char *argv[])
{
    ros::init(argc,argv,"tracker_node");
    ros::NodeHandle nh("~");

    nh.param("track_gate_chi2", current_params.track_gate_chi2, 11.34);
    nh.param("track_confirm_hits", current_params.track_confirm_hits, 2);
    nh.param("track_lost_time", current_params.track_lost_time, 0.5);
    nh.param("track_max_tracks", current_params.track_max_tracks, 8);
    nh.param("track_R_xy", current_params.track_R_xy, 1e-4);
    nh.param("track_R_z", current_params.track_R_z, 0.02);
    nh.param("track_acc_var", current_params.track_acc_var, 6.0);
    nh.param("track_vel_var_max", current_params.track_vel_var_max, 0.50);
    nh.param("track_v_max", current_params.track_v_max, 5.0);
    nh.param("track_a_max", current_params.track_a_max, 10.0);
    nh.param("T", current_params.T, 0.1);
    nh.param("track_reacq_time", current_params.track_reacq_time, 1.0);
    nh.param("track_reacq_dist", current_params.track_reacq_dist, 0.8);
    nh.param("track_id_memory", current_params.track_id_memory, 4.0);
    nh.param<std::string>("tf_parent_frame", g_tf_frame, std::string("camera_optical_frame"));

    reconfigCallback(current_params, 0);
    dynamic_reconfigure::Server<ft_tracker::ArmorParamsConfig> server;
    dynamic_reconfigure::Server<ft_tracker::ArmorParamsConfig>::CallbackType f;
    f = boost::bind(&reconfigCallback, _1, _2);
    server.setCallback(f);
    server.updateConfig(current_params);//用YAML覆盖CFG默认值
    
    std::string detection_topic;
    nh.param<std::string> ("detection_topic", detection_topic, "/ft/detection");
    int queue_size = 10;
    nh.param("queue_size", queue_size, 10);

    ros::Subscriber sub = nh.subscribe(detection_topic,queue_size,trackCallback);
    track_pub = nh.advertise<ft_tracker::TrackArray> ("/ft/track",10);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/ft/debug/marker",10);

    ros::spin();
    return 0;
}

void trackCallback(const ft_detection::DetectionFrame::ConstPtr& msg)
{
    const double stamp = msg->header.stamp.toSec();
    if(stamp <= 0.0) return;
    static double last_stamp = -1.0;
    const bool time_reset = (last_stamp > 0.0 && stamp < last_stamp - 0.5);
    if(last_stamp > 0.0 && stamp < last_stamp - 1e-6){
        ROS_WARN_THROTTLE(2.0,"tracker: time jumped back (%.3f -> %.3f), tracks reset", last_stamp, stamp);
    }
    last_stamp = stamp;
    //消息变成观测数据
    std::vector<ft_tracker::Measurement> meas;
    meas.reserve(msg->detections.size());
    for(const auto& d : msg->detections){
        ft_tracker::Measurement m;
        float cx = 0.f, cy = 0.f;
        for(int i = 0;i < 4;i++){
            cx += d.corners[i].x;
            cy += d.corners[i].y;
        }
        m.center = cv::Point2f(cx * 0.25f, cy * 0.25f);
        //板宽 = 左上到右上的距离（像素距离）
        m.width = std::hypot(static_cast<double>(d.corners[1].x - d.corners[0].x),
                             static_cast<double>(d.corners[1].y - d.corners[0].y));
        if(m.width < 1.0) continue;//退化检测（板宽最小阈值）
        
        m.pos[0] = d.pose.position.x;
        m.pos[1] = d.pose.position.y;
        m.pos[2] = d.pose.position.z;
        m.q[0] = d.pose.orientation.x;
        m.q[1] = d.pose.orientation.y;
        m.q[2] = d.pose.orientation.z;
        m.q[3] = d.pose.orientation.w;
        m.is_red = d.is_red;
        m.is_large = d.is_large_armor;
        
        if (!std::isfinite(m.pos[0]) || !std::isfinite(m.pos[1]) ||
            !std::isfinite(m.pos[2]))continue;
        meas.push_back(m);
    }

    //更新
    const std::vector<ft_tracker::TrackState> tracks = tracker.update(meas,stamp);

    //目标层 id 管理，用于重连
    static int locked_track = -1;//当前锁定的内部 track id
    static int tgt_id = -1;//对外发布的 id
    static int next_tgt_id = 1;
    static bool tgt_alive = false;
    static double tgt_pos[3] = {0.0, 0.0, 0.0};
    static double tgt_vel[3] = {0.0, 0.0, 0.0};
    static double tgt_seen = 0.0;//最后一次"真实观测"的时刻

    if(time_reset){//重置目标层
        tgt_alive = false;  tgt_id = -1;  locked_track = -1;
        next_tgt_id = 1;    tgt_seen = 0.0;
        ROS_WARN("tracker: time reset, target layer cleared");
    }

    const ft_tracker::TrackState* locked = nullptr;
    for(const auto& t : tracks){
        if(static_cast<int>(t.id) == locked_track){
            locked = &t;
            break;
        }
    }

    const double age = (tgt_alive && stamp > tgt_seen) ? (stamp - tgt_seen) : 0.0;
    double pred[3] = {tgt_pos[0], tgt_pos[1], tgt_pos[2]};
    if(tgt_alive){
        double d[3] = {tgt_vel[0] * age, tgt_vel[1] * age, tgt_vel[2] * age};
        const double dn = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);//外推距离
        const double k = (dn > 2.0 && dn > 1e-9) ? (2.0 / dn) : 1.0;//限制最多2m
        for(int i = 0; i < 3; ++i) pred[i] = tgt_pos[i] + d[i] * k;
    }

    if(!locked && tgt_alive && age <= g_id_memory){
        //找一个能接回来的轨迹
        const double v = std::sqrt(tgt_vel[0]*tgt_vel[0] + tgt_vel[1]*tgt_vel[1] + tgt_vel[2]*tgt_vel[2]);
        //允许的偏差封顶4m
        const double gate = std::min(g_reacq_dist + 0.5 * v * age, 4.0);
        const ft_tracker::TrackState* cand = nullptr;
        double best = 1e9;
        for(const auto& t : tracks){
            if(!t.confirmed) continue;
            if(static_cast<int>(t.id) == locked_track) continue;
            if(!plausiblePos(t.pos)) continue;
            const double ax = t.pos[0]-pred[0], ay = t.pos[1]-pred[1], az = t.pos[2]-pred[2];
            const double bx = t.pos[0]-tgt_pos[0], by = t.pos[1]-tgt_pos[1], bz = t.pos[2]-tgt_pos[2];
            const double dp = std::sqrt(ax*ax + ay*ay + az*az);//轨迹到预测
            const double dl = std::sqrt(bx*bx + by*by + bz*bz);//轨迹到上一次看见
            const double d = std::min(dp, dl);
            if(d < best){ best = d; cand = &t; }
        }
        
        // // [诊断] 接不上时打一行：有几条轨迹、最近的离多远、门限多少。
        // //   tracks=0        → 那几秒根本没有已确认轨迹（检测太散，凑不够 2 帧）
        // //   best>gate       → 有轨迹但太远（外推漂了，或门限太小）
        // if(!cand || best > gate){
        //     ROS_INFO_THROTTLE(1.0, "reacq miss: age=%.2fs best=%.2fm gate=%.2fm tracks=%zu",
        //                       age, cand ? best : -1.0, gate, tracks.size());
        // }
        if(cand && best <= gate){ locked = cand; locked_track = static_cast<int>(cand->id); }
    }

    if(locked){//锁定的讨论
        if(!tgt_alive){//船新目标，发新号
            tgt_alive = true;
            tgt_id = next_tgt_id++;
        }
        if(locked->tracking && plausiblePos(locked->pos)){//在跟踪
            for(int i = 0; i < 3; ++i){
                tgt_pos[i] = locked->pos[i];
                if(locked->valid) tgt_vel[i] = locked->vel[i];
            }
            tgt_seen = stamp;
        }
    }else if(tgt_alive && age > g_id_memory){//目标丢失且记忆超时
        tgt_alive = false;
        tgt_id = -1;
        locked_track = -1;
    }else if(!tgt_alive && !tracks.empty()){//发现船新目标
        int best_hits = -1;//根据命中数最多来锁定
        for(const auto& t : tracks){
            if(!t.confirmed) continue;
            if(!plausiblePos(t.pos)) continue;
            if(t.hits > best_hits){ best_hits = t.hits; locked = &t; }
        }
        if(locked){
            locked_track = static_cast<int>(locked->id);
            tgt_alive = true;
            tgt_id = next_tgt_id++;
            for(int i = 0; i < 3; ++i){
                tgt_pos[i] = locked->pos[i];
                tgt_vel[i] = locked->vel[i];
            }
            tgt_seen = stamp;
        }
    }

    //打包要发送的信息
    ft_tracker::TrackArray out;
    out.header = msg->header;
    out.header.frame_id = g_tf_frame;// [改] 时间戳沿用图像，坐标系强制成可信的那一帧
    if(locked){//正常跟踪
        const auto& t = *locked;
        ft_tracker::Track tr;
        tr.id = static_cast<uint32_t>(tgt_id >= 0 ? tgt_id : static_cast<int>(t.id));
        tr.lost_time = static_cast<float>(t.lost_time);
        tr.tracking = t.tracking;
        tr.Pose.position.x = t.pos[0];
        tr.Pose.position.y = t.pos[1];
        tr.Pose.position.z = t.pos[2];
        tr.Pose.orientation.x = t.q[0];
        tr.Pose.orientation.y = t.q[1];
        tr.Pose.orientation.z = t.q[2];
        tr.Pose.orientation.w = t.q[3];
        tr.velocity.x = t.vel[0];
        tr.velocity.y = t.vel[1];
        tr.velocity.z = t.vel[2];
        tr.valid = t.valid;
        tr.position_predict.x = t.predict_pos[0];
        tr.position_predict.y = t.predict_pos[1];
        tr.position_predict.z = t.predict_pos[2];
        out.tracks.push_back(tr);
    }else if(tgt_alive){//目标存活但轨迹丢失，保留外推位置和角度
        ft_tracker::Track tr;
        tr.id = static_cast<uint32_t>(tgt_id >= 0 ? tgt_id : 0);
        tr.lost_time = static_cast<float>(stamp - tgt_seen);
        tr.tracking = false;
        tr.valid = false;
        tr.Pose.position.x = pred[0];
        tr.Pose.position.y = pred[1];
        tr.Pose.position.z = pred[2];
        tr.Pose.orientation.w = 1.0;//外推不做姿态，给单位四元数
        tr.velocity.x = tgt_vel[0];
        tr.velocity.y = tgt_vel[1];
        tr.velocity.z = tgt_vel[2];
        tr.position_predict.x = pred[0] + tgt_vel[0] * params.T;
        tr.position_predict.y = pred[1] + tgt_vel[1] * params.T;
        tr.position_predict.z = pred[2] + tgt_vel[2] * params.T;
        out.tracks.push_back(tr);
    }
    track_pub.publish(out);

    //发布Rviz预测位置
    visualization_msgs::MarkerArray arr;
    const double T = params.T;
    for(const auto& t : out.tracks){
        const bool real = t.tracking && t.valid;//判断是预测还是外推
        const double px = t.Pose.position.x;
        const double py = t.Pose.position.y;
        const double pz = t.Pose.position.z;
        geometry_msgs::Point p0, p1;
        p0.x = px;
        p0.y = py;
        p0.z = pz;
        p1.x = px + t.velocity.x * T;
        p1.y = py + t.velocity.y * T;
        p1.z = pz + t.velocity.z * T;
        if(real){//画箭头
        visualization_msgs::Marker arrow;
        arrow.header = out.header;
        arrow.ns = "predict_arrow";
        arrow.id = static_cast<int>(t.id);
        arrow.type = visualization_msgs::Marker::ARROW;
        arrow.action = visualization_msgs::Marker::ADD;
        arrow.pose.orientation.w = 1.0;
        arrow.scale.x = 0.015;//杆直径
        arrow.scale.y = 0.030;//箭头直径
        arrow.scale.z = 0.040;//箭头长度
        arrow.color.r = 1.0f;
        arrow.color.g = 0.2f;
        arrow.color.b = 0.2f;
        arrow.color.a = 1.0f;
        arrow.lifetime = ros::Duration(0.2);

        arrow.points.push_back(p0);
        arrow.points.push_back(p1);
        arr.markers.push_back(arrow);
        }
        visualization_msgs::Marker dot;
        dot.header = out.header;
        dot.ns = "predict_point";
        dot.id = static_cast<int>(t.id);
        dot.type = visualization_msgs::Marker::SPHERE;
        dot.action = visualization_msgs::Marker::ADD;
        dot.pose.position = p1;
        dot.pose.orientation.w = 1.0;
        dot.scale.x = dot.scale.y = dot.scale.z = 0.08;
        if(real){//真实预测（青色球）
            dot.color.r = 0.0f;
            dot.color.g = 1.0f;
            dot.color.b = 1.0f;
            dot.color.a = 0.85f;
        }else{//目标丢失外推预测（灰色球）
            dot.color.r = 0.6f;
            dot.color.g = 0.6f;
            dot.color.b = 0.6f;
            dot.color.a = 0.55f;
        }
        dot.lifetime = ros::Duration(0.2);
        arr.markers.push_back(dot);
    }
    marker_pub.publish(arr);

    // // [改] 日志只打印"锁定目标"，而且带上 id/hits。
    // //   原来的写法是遍历 tracks 取第一条 tracking 的轨迹，多目标时会打到另一块板的新轨迹上，
    // //   看起来就像"sigv 突然变大"，其实根本不是同一条轨迹。
    // //   判读方法：hits 小（刚建）→ sigv 大是正常的收敛过程；hits 大还大 → 才是参数问题。
    // if(locked){
    //     const double speed = std::sqrt(locked->vel[0]*locked->vel[0] +
    //                                    locked->vel[1]*locked->vel[1] +
    //                                    locked->vel[2]*locked->vel[2]);
    //     ROS_INFO_THROTTLE(0.5, "id=%d(t%d) hits=%d made=%zu |v|=%5.2f sigv=%.2f valid=%d trk=%d raw=%zu meas=%zu",
    //                       tgt_id, locked_track, locked->hits, tracker.totalCreated(), speed,
    //                       locked->sigma_v, static_cast<int>(locked->valid),
    //                       static_cast<int>(locked->tracking),
    //                       msg->detections.size(), meas.size());
    // }else if(tgt_alive){
    //     // 断档期：轨迹已删。
    //     //   age<=reacq_time  → 正在发外推位置（PREDICT）
    //     //   age> reacq_time  → 已停止外推发布，但 id 还在记忆里（MEMORY），等轨迹回来接
    //     const double speed = std::sqrt(tgt_vel[0]*tgt_vel[0] + tgt_vel[1]*tgt_vel[1] + tgt_vel[2]*tgt_vel[2]);
    //     const double a = stamp - tgt_seen;
    //     if(a <= g_reacq_time)
    //         ROS_INFO_THROTTLE(0.5, "id=%d PREDICT age=%.2fs |v|=%5.2f raw=%zu meas=%zu",
    //                           tgt_id, a, speed, msg->detections.size(), meas.size());
    //     else
    //         ROS_INFO_THROTTLE(0.5, "id=%d MEMORY  age=%.2fs (no output, waiting to re-attach)", tgt_id, a);
    // }else{
    //     ROS_INFO_THROTTLE(0.5, "no target, made=%zu", tracker.totalCreated());
    // }

    
    //     // 诊断：pub 里第一条有观测的轨迹的速度
    // double speed = 0.0;
    // bool   vld   = false;
    // for (const auto& t : tracks) {
    //     if (!t.tracking) continue;
    //     speed = std::sqrt(t.vel[0] * t.vel[0] + t.vel[1] * t.vel[1] + t.vel[2] * t.vel[2]);
    //     // sigma = 0.0;
    //     vld   = t.valid;
    //     break;
    // }
    // std::string id_str;
    // for (const auto& t : out.tracks) {
    //     id_str += std::to_string(t.id);
    //     id_str += t.tracking ? "*" : "~";
    //     id_str += " ";
    // }
    // ROS_INFO_THROTTLE(0.2, "t=%.3f raw=%zu meas=%zu pub=%zu made=%zu "
    //                   "sig=%.2f ok=%d [%s]",
    //                   stamp, msg->detections.size(), meas.size(), out.tracks.size(),
    //                   tracker.totalCreated(), speed, static_cast<int>(vld),
    //                   id_str.c_str());
}

void reconfigCallback(ft_tracker::ArmorParamsConfig &config, uint32_t level)
{
    current_params = config;

    params.gate         = config.track_gate_chi2;
    params.confirm_hits = config.track_confirm_hits;
    params.lost_time    = config.track_lost_time;
    params.max_tracks   = config.track_max_tracks;
    params.R_xy         = config.track_R_xy;
    params.R_z          = config.track_R_z;
    params.acc_var      = config.track_acc_var;
    params.vel_var_max  = config.track_vel_var_max;
    params.v_max        = config.track_v_max;
    params.a_max        = config.track_a_max;
    params.dv_step      = config.track_dv_step;
    g_reacq_time        = config.track_reacq_time;
    g_reacq_dist        = config.track_reacq_dist;
    g_id_memory         = config.track_id_memory;
    params.T            = config.T;
    tracker.setParams(params);
}