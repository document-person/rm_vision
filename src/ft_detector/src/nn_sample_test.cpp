// 用法:
//   ./devel/lib/ft_detector/nn_sample_test <model.onnx> <image1> [image2 ...]
//   调试图输出到 /tmp/nn_out_<序号>.jpg

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "ft_detector/armor_detector_nn.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "nn_sample_test", ros::init_options::NoRosout);
    ros::Time::init();// 离线工具：初始化 ros::Time，否则 ROS_*_THROTTLE 会抛 TimeNotInitialized
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 3) {
        ROS_ERROR("usage: nn_sample_test <model.onnx> <image1> [image2 ...]");
        return 1;
    }

    ArmorDetectorNN::Params params;
    params.model_path = argv[1];

    ArmorDetectorNN detector;
    if (!detector.init(params)) {
        ROS_ERROR("NN init failed");
        return 2;
    }

    for (int i = 2; i < argc; ++i) {
        cv::Mat img = cv::imread(argv[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            ROS_WARN("cannot read %s", argv[i]);
            continue;
        }

        const std::string name = argv[i];
        const size_t slash = name.find_last_of('/');
        ROS_INFO("===== %s  %dx%d =====",
                 (slash == std::string::npos ? name : name.substr(slash + 1)).c_str(),
                 img.cols, img.rows);

        const std::vector<Armor> armors = detector.detect(img);
        ROS_INFO("detections=%zu  total=%.2f ms", armors.size(), detector.lastTotalMs());

        for (size_t k = 0; k < armors.size(); ++k) {
            const Armor& a = armors[k];
            ROS_INFO("  #%zu  %-3s %-5s number=%d  conf=%.3f  box=(%d,%d,%d,%d)",
                     k, a.color.c_str(), a.type.c_str(), a.number, a.confidence,
                     a.bounding_box.x, a.bounding_box.y,
                     a.bounding_box.width, a.bounding_box.height);
        }

        detector.drawArmors(img, armors);
        const std::string out = "/tmp/nn_out_" + std::to_string(i) + ".jpg";
        cv::imwrite(out, img);
        if (!armors.empty()) ROS_INFO("debug image -> %s", out.c_str());
    }
    return 0;
}
