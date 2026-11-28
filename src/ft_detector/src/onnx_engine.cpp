#include "ft_detector/onnx_engine.h"

#include <ros/ros.h>

#include <array>
#include <chrono>
#include <cstdio>

namespace {
// 辅助函数：计算距离 t0 经过了多少毫秒
double msSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

}

OnnxEngine::~OnnxEngine() = default;

bool OnnxEngine::load(const std::string& model_path, int intra_threads)
{
    ready_ = false;
#ifdef FT_USE_ONNXRUNTIME
    try{
        //创建 ONNX 运行环境，指定日志级别为 WARNING
        env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ft_nn"));
        //配置会话参数
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(intra_threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);//图优化
        //加载模型，创建 session
        session_.reset(new Ort::Session(*env_, model_path.c_str(), so));
        //获取模型输入输出节点的名称
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_->GetInputNameAllocated(0, alloc).get();
        out_name_ = session_->GetOutputNameAllocated(0, alloc).get();
        //获取模型输入节点的形状信息，用于日志打印
        const auto in_type = session_->GetInputTypeInfo(0);
        const auto in_info = in_type.GetTensorTypeAndShapeInfo();//获取张量形状视图

        version_ = Ort::GetVersionString();
        std::string in_shape;
        for(int64_t d : in_info.GetShape()) 
        {
            in_shape += std::to_string(d) + " ";
        }
        ROS_INFO("[nn] ort=%s in=%s [%s] out=%s",
                 version_.c_str(), in_name_.c_str(), in_shape.c_str(), out_name_.c_str());
        ready_ = true;
        return true;
    }catch(const Ort::Exception& e){
        ROS_ERROR("[nn] onnxruntime load failed: %s", e.what());
        return false;
    }
    #else
    (void)intra_threads;
    ROS_ERROR("[nn] Dont's use OnnxRuntime,can't find %s", model_path.c_str());
    ROS_ERROR("[nn] Please checks ORT_ROOT in Cmake,is or not point to include/onnxruntime_cxx_api.h");
    return false;
#endif
}

bool OnnxEngine::infer(const cv::Mat& blob, cv::Mat& output)
{
    if(!ready_) return false;
    if(blob.empty() || blob.dims != 4 || !blob.isContinuous()){//张量是否合理
        ROS_WARN_THROTTLE(2.0, "[nn] blob must be 4 dimension NCHW continuous memory");
        return false;
    }
    
#ifdef FT_USE_ONNXRUNTIME
    const auto t0 = std::chrono::steady_clock::now();
    //构造张量形状
    std::array<int64_t,4> shape{blob.size[0], blob.size[1], blob.size[2], blob.size[3]};
    //创建 CPU 内存信息
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //使用 cv::Mat 的底层数据指针，创建 ORT 输入张量（注意这里只引用，不拷贝）
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float*>(blob.data), blob.total(), shape.data(), 4);
    const char* in_names[] = {in_name_.c_str()};
    const char* out_names[] = {out_name_.c_str()};
    //执行推理
    std::vector<Ort::Value> outs;
    try{
        outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in,1,out_names, 1);
    }catch(const Ort::Exception& e){
        ROS_ERROR_THROTTLE(2.0, "[nn] forward failed: %s", e.what());
        return false;
    }
    //解析输出形状并保存
    const auto info = outs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> dims = info.GetShape();
    out_shape_.assign(dims.begin(), dims.end());
    const std::vector<int> sizes(dims.begin(), dims.end());
    //将 ORT 的输出转换成 OpenCV Mat 返回
    // 极其重要：ORT 的输出缓冲会在函数结束时释放，所以必须用 clone() 深拷贝一份！
    output = cv::Mat(static_cast<int>(sizes.size()), sizes.data(), CV_32F,
                     const_cast<float*>(outs[0].GetTensorData<float>())).clone();
    last_infer_ms_ = msSince(t0);
    return !output.empty();
#else
    (void)output;
    return false;
#endif
}