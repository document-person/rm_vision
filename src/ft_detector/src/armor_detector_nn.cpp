#include "ft_detector/armor_detector_nn.h"

#include <ros/package.h>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace{
double msSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

// 参数越界保护：返回 clamp 后的值，被改动时打印一次。
// 这些值来自 YAML，不能直接信（例如 nn_max_detections=-1 会让 resize 拿到一个天文数字）。
float clampF(const char* name, float v, float lo, float hi)
{
    const float c = std::min(std::max(v, lo), hi);
    if(c != v){
        ROS_WARN("[nn] %s=%.3f out of [%.2f,%.2f], clamped to %.3f", name, v, lo, hi, c);
    }
    return c;
}
int clampI(const char* name, int v, int lo, int hi)
{
    const int c = std::min(std::max(v, lo), hi);
    if(c != v){
        ROS_WARN("[nn] %s=%d out of [%d,%d], clamped to %d", name, v, lo, hi, c);
    }
    return c;
}
}

namespace ft_nn{
std::string resolveModelPath(const std::string& path)
{
    if(path.empty()){
        return ros::package::getPath("ft_detector") + "/model/armor.onnx";
    }
    const std::string prefix = "package://";
    if(path.compare(0,prefix.size(), prefix) == 0){
        const std::string rest = path.substr(prefix.size());
        const size_t slash = rest.find('/');
        if(slash == std::string::npos) return ros::package::getPath(rest);
        return ros::package::getPath(rest.substr(0,slash)) + rest.substr(slash);
    }
    return path;
}   
}

bool ArmorDetectorNN::init(const Params& params)
{
    params_ = params;
    ready_ = false;
    //输入宽高
    if(params_.input_w <= 0 || params_.input_h <= 0){
        ROS_ERROR("[nn] bad input size %dx%d", params_.input_w, params_.input_h);
        return false;
    }
    //把来自 YAML 的阈值/数量参数收敛到安全范围
    params_.color_thresh   = clampF("nn_color_thresh",   params_.color_thresh,   0.01f, 1.0f);
    params_.number_thresh  = clampF("nn_number_thresh",  params_.number_thresh,  0.01f, 1.0f);
    params_.nms_iou        = clampF("nn_nms_iou",        params_.nms_iou,        0.0f,  1.0f);
    params_.pad_value      = clampI("nn_pad_value",      params_.pad_value,      0,     255);
    params_.max_detections = clampI("nn_max_detections", params_.max_detections, 1,     1000);
    params_.intra_threads  = clampI("nn_intra_threads",  params_.intra_threads,  1,     64);
    params_.debug_every_n  = clampI("nn_debug_every_n",  params_.debug_every_n,  1,     100000);
    if(params_.large_by_number.size() != 9){
        ROS_WARN("[nn] large_by_number's lenth should be 9,back to all small");
        params_.large_by_number.assign(9, 0);
    }

    const std::string path = ft_nn::resolveModelPath(params_.model_path);
    ROS_INFO("[nn] model path = %s", path.c_str());
    if(!engine_.load(path, params_.intra_threads)) return false;
    
    ready_ = true;
    ROS_INFO("[nn] ready: input=%dx%d pad=%d color=%.2f number=%.2f nms=%.2f",
             params_.input_w, params_.input_h, params_.pad_value,
             params_.color_thresh, params_.number_thresh, params_.nms_iou);
    return true;
}

bool ArmorDetectorNN::initFromRos(const ros::NodeHandle& nh)
{
    Params p = params_;// 保留代码默认值，YAML 只覆盖存在的键
    nh.param<std::string>("nn_model_path", p.model_path, p.model_path);
    nh.param("nn_input_width", p.input_w, p.input_w);
    nh.param("nn_input_height", p.input_h, p.input_h);
    nh.param("nn_pad_value", p.pad_value, p.pad_value);
    nh.param("nn_color_thresh", p.color_thresh, p.color_thresh);
    nh.param("nn_number_thresh", p.number_thresh, p.number_thresh);
    nh.param("nn_nms_iou", p.nms_iou, p.nms_iou);
    nh.param("nn_max_detections", p.max_detections, p.max_detections);
    nh.param("nn_intra_threads", p.intra_threads, p.intra_threads);
    nh.param("nn_verbose_output", p.verbose_output, p.verbose_output);
    nh.param("nn_debug_every_n", p.debug_every_n, p.debug_every_n);
    if (!nh.getParam("nn_large_by_number", p.large_by_number) ||
        p.large_by_number.size() != 9) {
        ROS_WARN_THROTTLE(1.0,"[nn] nn_large_by_number lost or longth != 9,all back to small");
        p.large_by_number.assign(9, 0);
    }
    return init(p);
}

cv::Mat ArmorDetectorNN::preprocess(const cv::Mat& bgr, ft_nn::LetterboxInfo& lb) const
{
    const int in_w = params_.input_w;
    const int in_h = params_.input_h;
    //等比缩放
    lb.scale = std::min(in_w / static_cast<float>(bgr.cols),
                        in_h / static_cast<float>(bgr.rows));
    const int nw = std::max(1, static_cast<int>(std::lround(bgr.cols * lb.scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(bgr.rows * lb.scale)));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw,nh), 0, 0,cv::INTER_LINEAR);
    //居中补边
    lb.pad_x = (in_w - nw) / 2;
    lb.pad_y = (in_h - nh) / 2;
    lb.canvas = cv::Size(in_w, in_h);
    cv::Mat canvas(in_h, in_w, bgr.type(),
                   cv::Scalar(params_.pad_value, params_.pad_value, params_.pad_value));
    resized.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));
    
    //转张量
    return cv::dnn::blobFromImage(canvas, 1.0/ 255.0, cv::Size(in_w,in_h),
                                  cv::Scalar(), params_.to_rgb, false, CV_32F);
}

std::vector<Armor> ArmorDetectorNN::decode(const cv::Mat& output,
                    const ft_nn::LetterboxInfo& lb, const cv::Size& src_size) const
{
    std::vector<Armor> armors;
    if(output.empty() || output.dims != 3 || !output.isContinuous()) return armors;
    if(output.size[0] != 1 || output.size[1] < 21) return armors;

    const int C = output.size[1];//21
    const int N = output.size[2];//6300
    cv::Mat m = output.reshape(1,C);//转换为矩阵，一列一个候选
    const float* p = m.ptr<float>(0);// 取值 p[c * N + i]
    
    auto mapClip = [&](const cv::Point2f& q){
        cv::Point2f s = lb.toSource(q);
        s.x = std::min(std::max(s.x, 0.0f), static_cast<float>(src_size.width - 1));
        s.y = std::min(std::max(s.y, 0.0f), static_cast<float>(src_size.height - 1));
        return s;
    };
    
    for(int i = 0; i < N; ++i){
        //颜色：c0=蓝 c1=红 c2=熄灭 c3=其他(忽略)
        int ci = 0;
        float cs = p[i];
        for(int c = 1; c < 3; ++c){
            const float v = p[c * N + i];
            if(v > cs){
                cs = v;
                ci = c;
            }
        }
        if(cs < params_.color_thresh) continue;
        if(ci == 2) continue;//熄灭的板子不要
        
        //编号
        int ni = 0;
        float ns = p[4 * N + i];
        for(int n = 1; n < 9; ++n){
            const float v = p[(4 + n) * N + i];
            if(v > ns){
                ns = v;
                ni = n;
            }
        }
        if(ns < params_.number_thresh) continue;
        
        //角点：ch13..20，模型顺序 左上、左下、右下、右上
        const cv::Point2f tl(p[13 * N + i], p[14 * N + i]);
        const cv::Point2f bl(p[15 * N + i], p[16 * N + i]);
        const cv::Point2f br(p[17 * N + i], p[18 * N + i]);
        const cv::Point2f tr(p[19 * N + i], p[20 * N + i]);
        
        //打包
        Armor a;
        a.corners = {mapClip(tl), mapClip(tr), mapClip(br), mapClip(bl)};
        a.bounding_box = cv::boundingRect(a.corners);
        a.color = (ci == 1) ? "Red" : "Blue";
        a.type = (params_.large_by_number[ni] != 0) ? "Big" : "Small";
        a.confidence = cs;//用颜色分数当置信度
        a.number = ni;
        armors.push_back(a);
    }

    std::sort(armors.begin(), armors.end(),
              [](const Armor& a, const Armor& b){return a.confidence > b.confidence;});
    return armors;
}

void ArmorDetectorNN::nms(std::vector<Armor>& armors, float iou_thresh)
{
    if(armors.size() < 2) return;
    std::vector<char> keep(armors.size(), 1);//标记数组，1表示保留，0表示丢弃
    for(size_t i = 0; i < armors.size(); ++i){
        if(!keep[i]) continue;
        for(size_t j = i+1; j < armors.size(); ++j){
            if(!keep[j]) continue;
            //按颜色分组
            if(armors[i].color != armors[j].color) continue;
            const cv::Rect& a = armors[i].bounding_box;
            const cv::Rect& b = armors[j].bounding_box;
            const int inter = (a & b).area();//交集
            const int uni = a.area() + b.area() - inter;//并集
            const float iou = (uni > 0) ? static_cast<float>(inter) / uni : 0.0f;
            if(iou > iou_thresh) keep[j] = 0;
        }
    }
    std::vector<Armor> out;
    out.reserve(armors.size());
    for(size_t i = 0; i < armors.size(); ++i){
        if(keep[i]){
            out.push_back(armors[i]);
        }
    }
    armors.swap(out);
}

std::vector<Armor> ArmorDetectorNN::detect(const cv::Mat& bgr)
{
    std::vector<Armor> armors;
    if(!ready_ || bgr.empty()) return armors;
    //预处理
    const auto t0 = std::chrono::steady_clock::now();
    ft_nn::LetterboxInfo lb;
    cv::Mat blob = preprocess(bgr,lb);
    last_lb_ = lb;
    last_pre_ms_ = msSince(t0);
    if(blob.empty()) return armors;
    //模型推理
    cv::Mat output;
    if(!engine_.infer(blob,output)) return armors;
    //可选调试
    const auto t1 = std::chrono::steady_clock::now();
    if(params_.verbose_output &&
       frame_count_ % std::max(1, params_.debug_every_n) == 0){
        logOutoutStats(output);
    }
    //解码 + nms去重
    armors = decode(output, lb, bgr.size());
    if(armors.size() > 1) nms(armors, params_.nms_iou);
    if(static_cast<int>(armors.size()) > params_.max_detections){
        armors.resize(static_cast<size_t>(params_.max_detections));
    }
    last_post_ms_ = msSince(t1);
    frame_count_ += 1;
    last_total_ms_ = last_pre_ms_ + engine_.lastInferMs() + last_post_ms_;
    total_hits_.push_back(last_total_ms_);
    if(total_hits_.size() > 100) total_hits_.pop_front();
    if(frame_count_ % std::max(1, params_.debug_every_n) == 0) logTiming();
    
    return armors;
}

void ArmorDetectorNN::logOutoutStats(const cv::Mat& output) const
{
    if(output.empty()) return;
    
    std::string shape;
    for(int i = 0; i < output.dims; ++i){
        shape += (i ? "x" : "") + std::to_string(output.size[i]);
    }
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(output.reshape(1,1), &lo, &hi);//可用于查看模型的质量
    ROS_INFO_THROTTLE(1.0,"[nn] output [%s] min=%.3f max=%.3f", shape.c_str(), lo, hi);
    
    if(output.dims != 3 || !output.isContinuous() || output.size[1] < 21) return;
    cv::Mat m = output.reshape(1, output.size[1]);// C x N
    auto rangeOf = [&m](int r0, int r1, double& a, double& b){
        cv::Mat block = m.rowRange(r0,r1);
        if(block.rows > 1){
            cv::Mat mx;
            cv::reduce(block, mx, 0, cv::REDUCE_MAX);
            cv::minMaxLoc(mx, &a, &b);
        }else{
            cv::minMaxLoc(block, &a, &b);
        }   
    };
    double c0 = 0, c1 = 0, n0 = 0, n1 = 0, k0 = 0, k1 = 0;
    rangeOf(0, 3, c0, c1);//颜色
    rangeOf(4, 13, n0, n1);//编号
    rangeOf(13, 21, k0, k1);//角点
    ROS_INFO_THROTTLE(1.0,"[nn] color=[%.3f,%.3f] number=[%.3f,%.3f] corner_px=[%.1f,%.1f]",
                      c0, c1, n0, n1, k0, k1);
}

void ArmorDetectorNN::logTiming()
{
    if(total_hits_.empty()) return;
    
    double sum = 0.0;
    for(double v : total_hits_){
        sum += v;
    }
    const double avg = sum / static_cast<double>(total_hits_.size());
    
    std::vector<double> v(total_hits_.begin(), total_hits_.end());
    std::sort(v.begin(), v.end());
    const double p95 = v[static_cast<size_t>(0.95 * (v.size() - 1))];
    ROS_INFO_THROTTLE(1.0,"[nn] ms: pre=%.2f infer=%.2f post=%.2f total=%.2f | avg=%.2f p95=%.2f (n=%zu)",
            last_pre_ms_, engine_.lastInferMs(), last_post_ms_, last_total_ms_,
            avg, p95, v.size());
}

void ArmorDetectorNN::drawArmors(cv::Mat& image, const std::vector<Armor>& armors) const
{
    for(size_t i = 0; i < armors.size(); ++i){
        const Armor& a = armors[i];
        if(a.corners.size() != 4) continue;
        
        const cv::Scalar col = (a.color == "Blue") ? cv::Scalar(255,0,0) :
                                                     cv::Scalar(0,0,255);
        std::vector<cv::Point> poly;
        poly.reserve(4);
        for(const cv::Point2f& c : a.corners){
            poly.emplace_back(cvRound(c.x), cvRound(c.y));
        }
        cv::polylines(image, poly, true, col, 2);
        
        // 角点 0..3：肉眼确认就是 左上 右上 右下 左下
        for(size_t k = 0; k < 4; ++k){
            cv::circle(image, poly[k], 3, cv::Scalar(0,255,0), -1);
            cv::putText(image, std::to_string(k), poly[k] + cv::Point(4, -4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,255), 1);
        }
        
        const std::string label = cv::format("NN %s %s #%d %.2f", a.color.c_str(),
                                             a.type.c_str(), a.number, a.confidence);
         cv::putText(image, label,
                    cv::Point(a.bounding_box.x, std::max(12, a.bounding_box.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }
}