#include "ft_tracker/tracker.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ft_tracker{

void Tracker::reset()
{
    tracks_.clear();
    next_id = 1;
    last_stamp_ = 0.0;
}
//初始化卡尔曼
void Tracker::initKalman(Track& tr, const Measurement& m)
{
    tr.kf.init(6,3,0,CV_64F);
    // 初始状态：位置是观测值，速度假设为0
    tr.kf.statePost = (cv::Mat_<double>(6,1) <<
                       m.pos[0], m.pos[1], m.pos[2], 0.0f, 0.0f, 0.0f);
    cv::setIdentity(tr.kf.measurementMatrix);//测量矩阵 H：只提取前三个位置
    cv::setIdentity(tr.kf.transitionMatrix);//状态转移矩阵 F：初始为单位阵
    // 测量噪声 R
    tr.kf.measurementNoiseCov.setTo(0.0);
    tr.kf.measurementNoiseCov.at<double>(0,0) = params_.R_xy;
    tr.kf.measurementNoiseCov.at<double>(1,1) = params_.R_xy;
    tr.kf.measurementNoiseCov.at<double>(2,2) = params_.R_z;
    //协方差矩阵
    tr.kf.errorCovPost.setTo(0.0);
    tr.kf.errorCovPost.at<double>(0,0) = params_.R_xy;
    tr.kf.errorCovPost.at<double>(1,1) = params_.R_xy;
    tr.kf.errorCovPost.at<double>(2,2) = params_.R_z;
    tr.kf.errorCovPost.at<double>(3,3) = params_.v0_var;
    tr.kf.errorCovPost.at<double>(4,4) = params_.v0_var;
    tr.kf.errorCovPost.at<double>(5,5) = params_.v0_var;

    tr.kf_init_ready = true;
}
void Tracker::predictKalman(Track& tr, double dt)
{
    if(dt <= 0.0) return;
    // 1. 动态更新 F 矩阵 (状态转移矩阵)
    tr.kf.transitionMatrix.at<double>(0,3) = dt;
    tr.kf.transitionMatrix.at<double>(1,4) = dt;
    tr.kf.transitionMatrix.at<double>(2,5) = dt;
    // 2. 动态计算 Q 矩阵 (白噪声加速度模型)
    double dt2 = dt * dt;
    double q_p_p = params_.acc_var * dt2 * dt2 / 4.0;//位置与位置
    double q_p_v = params_.acc_var * dt2 * dt / 2.0;//位置与速度的交叉项
    const double dq = std::min(dt, 0.10);//Q 用限幅后的 dt
    double q_v_v = params_.acc_var * dq * dq;//速度与速度
    
    cv::Mat& Q = tr.kf.processNoiseCov;
    Q.setTo(0.0);
    for(int i = 0; i < 3; ++i){
        Q.at<double>(i,i) = q_p_p;
        Q.at<double>(i,i+3) = q_p_v;
        Q.at<double>(i+3,i) = q_p_v;
        Q.at<double>(i+3,i+3) = q_v_v;
    }
    //3、预测
    tr.kf.predict();
}

//传入位置和速度信息，超速就做速度限幅
void Tracker::clampVelocity(cv::Mat& state) const
{
    const double vx = state.at<double>(3);
    const double vy = state.at<double>(4);
    const double vz = state.at<double>(5);
    const double sp = std::sqrt(vx*vx + vy*vy + vz*vz);
    if(sp > params_.v_max && sp > 1e-9){
        const double k = params_.v_max / sp;
        state.at<double>(3) = vx * k;
        state.at<double>(4) = vy * k;
        state.at<double>(5) = vz * k;
    }
}

//限制加速度
void Tracker::clampVelocityDelta(cv::Mat& post, const cv::Mat& pre, double dt) const
{
    if(dt <= 0.0) return;
    const double dvx = post.at<double>(3) - pre.at<double>(3);
    const double dvy = post.at<double>(4) - pre.at<double>(4);
    const double dvz = post.at<double>(5) - pre.at<double>(5);
    const double dv  = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
    //两个上限取小
    const double dv_max = std::min(params_.a_max * dt, params_.dv_step);
    if(dv > dv_max && dv > 1e-9){
        const double k = dv_max / dv;
        post.at<double>(3) = pre.at<double>(3) + dvx * k;
        post.at<double>(4) = pre.at<double>(4) + dvy * k;
        post.at<double>(5) = pre.at<double>(5) + dvz * k;
    }
}

std::vector<TrackState> Tracker::update(const std::vector<Measurement>& meas,double stamp)
{
    if(last_stamp_ > 0.0 && stamp < last_stamp_ - 1e-9){
        if(last_stamp_ - stamp < 0.5) return last_out_;
        reset();
    }
    last_stamp_ = stamp;
    const int nt = static_cast<int> (tracks_.size());
    const int nm = static_cast<int> (meas.size());

    //1、预测
    for(auto& tr : tracks_){
        if(!tr.kf_init_ready) continue;
        double dt = stamp - tr.last_stamp;
        if(dt > 1e-4){
            //限制预测的dt
            if(dt > 0.3) dt = 0.3;
            predictKalman(tr, dt);
        }
    }
    
    // 2. 数据关联
    struct Edge
    {
        double cost;
        int t;
        int m;
    };
    std::vector<Edge> edges;
    edges.reserve(nt * nm);
    
    for(int t = 0; t < nt; ++t){
        const Track& tr = tracks_[t];
        if(!tr.kf_init_ready) continue;
        const cv::Mat& pred = tr.kf.statePre;//引用预测位置
        cv::Mat S = tr.kf.errorCovPre(cv::Rect(0, 0, 3, 3))
                  + tr.kf.measurementNoiseCov;//测量与观测误差相加得到总误差
        const double infl = params_.gate_sigma_m * params_.gate_sigma_m;
        for(int i = 0; i < 3; ++i) S.at<double>(i, i) += infl;//正则化
        const cv::Mat Sinv = S.inv();//求逆
        
        for(int m = 0; m < nm; ++m){
            const Measurement& me = meas[m];
            cv::Mat r = (cv::Mat_<double>(3,1) <<
                         me.pos[0] - pred.at<double>(0),
                         me.pos[1] - pred.at<double>(1),
                         me.pos[2] - pred.at<double>(2));//算残差
            const cv::Mat d2m = r.t() * Sinv * r;
            const double d2 = d2m.at<double>(0, 0);
            if(d2 > params_.gate) continue;

            double cost = d2;
            if(me.is_red != tr.last.is_red) cost += 50.0;//颜色不匹配重罚
            if(me.is_large != tr.last.is_large) cost += 5.0;//容忍极端情况出现
            edges.push_back({cost, t, m});
        }
    }
    std::sort(edges.begin(), edges.end(), 
              [](const Edge& a, const Edge& b) {return a.cost < b.cost;});
    //3、一对一贪心分配
    std::vector<int> track_of_meas(nm, -1);//测量对应的轨迹
    std::vector<char> track_used(nt, 0), meas_used(nm, 0);
    for(const auto& e : edges){
        if(track_used[e.t] || meas_used[e.m]) continue;
        track_used[e.t] = 1;
        meas_used[e.m] = 1;
        track_of_meas[e.m] = e.t;
    }
    //4、用观测更新轨迹
    for(int m = 0; m < nm; ++m){
        const int t = track_of_meas[m];
        if(t < 0) continue;
        Track& tr = tracks_[t];
        const double dt = stamp - tr.last_stamp;

        tr.last = meas[m];
        tr.last_stamp = stamp;
        tr.hits += 1;
        tr.misses = 0;
        tr.confirmed = tr.confirmed || (tr.hits >= params_.confirm_hits);

        //将观测值输入卡尔曼
        if(tr.kf_init_ready && dt >= 0.002){
            cv::Mat measurement = (cv::Mat_<double>(3,1) <<
                                   meas[m].pos[0], meas[m].pos[1], meas[m].pos[2]);
            tr.kf.correct(measurement);
            clampVelocityDelta(tr.kf.statePost, tr.kf.statePre, dt);
            clampVelocity(tr.kf.statePost);
        }
    }
    //5、没配上的轨迹老化
    for(int t = 0; t < nt; ++t){
        if(!track_used[t]) tracks_[t].misses += 1;
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(),tracks_.end(),
                       [&](const Track& tr){
                            if(tr.misses == 0) return false;// 本帧配上了，留着
                            return (stamp - tr.last_stamp) > params_.lost_time;
                       }),
        tracks_.end());
    
    //5、没配上的观测，新建轨迹
    for(int m = 0; m < nm; ++m){
        if(meas_used[m]) continue;
        if(nt >= params_.max_tracks) break;// 满了就不再新建
        Track tr;
        tr.id = next_id++;
        tr.last = meas[m];
        tr.last_stamp = stamp;
        tr.hits = 1;
        tr.confirmed = (params_.confirm_hits <= tr.hits);
        
        //初始化卡尔曼
        initKalman(tr,meas[m]);
        tracks_.push_back(tr);
    }
    //6、输出确认的轨迹
    std::vector<TrackState> out;
    out.reserve(tracks_.size());
    for(const auto& tr : tracks_){
        if(!tr.confirmed) continue;
        TrackState s;
        s.id = tr.id;
        s.center = tr.last.center;
        s.is_red = tr.last.is_red;
        s.width = tr.last.width;
        s.is_large = tr.last.is_large;
        s.confirmed = tr.confirmed;
        s.hits = tr.hits;
        s.tracking = (tr.misses == 0);
        s.lost_time = stamp - tr.last_stamp;

        //区分是有观测还是纯预测
        const cv::Mat& st = (tr.misses > 0) ? tr.kf.statePre : tr.kf.statePost;
        const cv::Mat& P = (tr.misses > 0) ? tr.kf.errorCovPre : tr.kf.errorCovPost;

        //从卡尔曼提取位置和速度
        if(tr.kf_init_ready){
            s.pos[0] = st.at<double>(0);
            s.pos[1] = st.at<double>(1);
            s.pos[2] = st.at<double>(2);
            s.vel[0] = st.at<double>(3);
            s.vel[1] = st.at<double>(4);
            s.vel[2] = st.at<double>(5);
            for(int i = 0; i < 3; ++i) s.predict_pos[i] = s.pos[i] + s.vel[i] * params_.T;
            double var_v = std::max(std::max(P.at<double>(3,3), P.at<double>(4,4)),
                                    P.at<double>(5,5));
            if(!(var_v > 0.0)) var_v = 0.0;
            s.sigma_v = std::sqrt(var_v);
            s.valid = (tr.hits >= params_.confirm_hits) && (var_v <= params_.vel_var_max);
            //速度不可信时不给提前量，直接回填当前位置
            if(!s.valid) for(int i = 0; i < 3; ++i) s.predict_pos[i] = s.pos[i];
        }else{//卡尔曼初始化失败，信息不可信，保持上一次的角度，但不预测位置
            for(int i = 0; i < 3; ++i){
                s.pos[i] = tr.last.pos[i];
                s.vel[i] = 0.0;
                s.predict_pos[i] = s.pos[i];
            }
            s.valid = false;
        }
        for(int i = 0; i < 4; ++i){
            s.q[i] = tr.last.q[i];
        }
        
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(),
            [](const TrackState& a, const TrackState& b) {return a.id < b.id;});
    last_out_ = out;//缓存这一帧的输出，时间戳回跳的那一帧直接复用它
    return out;
}

}