#ifndef FT_DETECTOR_ONNX_ENGINE_H
#define FT_DETECTOR_ONNX_ENGINE_H

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

//只有在安装了ORT时才编译
#ifdef FT_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// 只做两件事：加载模型、一次前向
class OnnxEngine
{
public:
    ~OnnxEngine();// 析构函数，用于安全释放 ONNX 内存

    bool load(const std::string& model_path, int intra_threads = 2);//加载模型
    bool ready() const { return ready_; }//判断引擎是否初始化成功

    // 核心推理接口。blob: 预处理好的图像张量（CV_32F，NCHW格式）。output: 模型输出结果
    bool infer(const cv::Mat& blob, cv::Mat& output);

    //获取模型的输出形状（例如：[1, 21, 6300]）
    const std::vector<int>& outputShape() const { return out_shape_; }
    //获取上一次推理所消耗的时间（毫秒），用于性能监控
    double lastInferMs() const { return last_infer_ms_; }
    //获取 ONNX Runtime 的版本号，用于日志打印
    const std::string& version() const { return version_; }

private:
#ifdef FT_USE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env_;//早到晚退
    std::unique_ptr<Ort::Session> session_;// 推理会话，绑定具体的模型文件
    std::string in_name_, out_name_;// 模型的输入节点名和输出节点名
#endif
    std::string version_;// ORT 版本号
    std::vector<int> out_shape_;// 输出张量的维度
    double last_infer_ms_ = 0.0;// 上一次推理的耗时（毫秒）
    bool ready_ = false;// 标记是否加载成功
};

#endif