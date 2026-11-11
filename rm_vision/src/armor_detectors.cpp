#include "rm_vision/armor_detector.h"
#include <algorithm>
#include <cmath>
#include <ros/ros.h>

ArmorDetector::ArmorDetector(){}

//颜色分割
cv::Mat ArmorDetector::getColorMask(const cv::Mat& hsv,bool is_blue)
{
    cv::Mat mask;
    if(is_blue){
        cv::inRange(hsv,cv::Scalar(95,90,120),cv::Scalar(135,255,255),mask);
    }else{
        cv::Mat mask1,mask2;
        cv::inRange(hsv,cv::Scalar(0,90,120),cv::Scalar(20,255,255),mask1);
        cv::inRange(hsv,cv::Scalar(150,90,120),cv::Scalar(180,255,255),mask2);
        cv::bitwise_or(mask1,mask2,mask);//把两段色阈拼起来
    }

    // 形态学膨胀操作：用一个 3x3 的矩形核，把断裂的灯条连起来，或者去掉细小的噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,kernel);
    //腐蚀后重新补全
    cv::dilate(mask,mask,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));

    return mask;
}
//找灯条
std::vector<LightBar> ArmorDetector::findLightBars(const cv::Mat& mask,const cv::Mat& hsv)
{
    std::vector<std::vector<cv::Point>> contours;//用二维数组去储存所有轮廓的坐标点
    // 找轮廓：RETR_EXTERNAL只找最外层轮廓，CHAIN_APPROX_SIMPLE只存拐点坐标
    cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> bars;
    for(const auto& c : contours){//遍历每轮找到的轮廓
        cv::RotatedRect rect = cv::minAreaRect(c);
        
        double w = std::min(rect.size.width,rect.size.height);//宽
        double h = std::max(rect.size.width,rect.size.height);//长
        //过滤不合尺寸的
        if(rect.size.area() < 20 || h/w < 1.5 || h/w > 20) continue;
        
        cv::Point2f pts[4];
        rect.points(pts);
        cv::Point2f e1 = pts[1] - pts[0];//注意得到的是向量
        cv::Point2f e2 = pts[2] - pts[1];
        // 取较长的那条边作为灯条的长轴方向
        cv::Point2f long_edge = (cv::norm(e1) >= cv::norm(e2)) ? e1 : e2;
        // 与竖直方向（y 轴负方向）的夹角：长边竖直时为 0
        double angle = std::atan2(long_edge.x, -long_edge.y) * 180.0 / CV_PI;
        // 灯条没有头尾，旋转 180° 看起来一样 → 把角度折到 [-90, 90)
        while (angle >=  90.0) angle -= 180.0;
        while (angle <  -90.0) angle += 180.0;
        if (std::abs(angle) > 45.0) continue;//检测横躺的灯条

        //统计这根灯条的平均亮度(V)，用于筛选掉地面反光的灯条
        cv::Rect br = rect.boundingRect() & cv::Rect(0,0,mask.cols,mask.rows);
        double mean_v = 0.0;
        if (br.area() > 0) {//判断灯条必须在图像里面
            mean_v = cv::mean(hsv(br), mask(br))[2];
        }

        // 用于调试：检查每根灯条的信息
        // cv::Rect br = rect.boundingRect();
        // br &= cv::Rect(0,0,mask.cols,mask.rows);
        // double fill = br.area() > 0 ? (cv::countNonZero(mask(br)) / (double)br.area()) : 0;
        // ROS_INFO_THROTTLE(1.0,"bar: w=%.1f len=%.1f ang=%.1f  fill=%.2f", w, h, angle, fill);

        //把合格的信息打包存入结构体，加入到结果列表中
        LightBar bar;
        bar.rect = rect;
        bar.angle = angle;
        bar.center = rect.center;
        bar.length = h;
        bar.width = w;
        bar.mean_v = mean_v;
        bars.push_back(bar);
    }
    return bars;
}
//配对装甲板
std::vector<Armor> ArmorDetector::matchArmors(
    const std::vector<LightBar>& bars,const std::string& color)
{// 核心逻辑：遍历两两灯条，判断角度是否互补、y轴高度差、距离比例，配对成装甲板
    //先找出本帧同色灯条里最亮的那根，作为判定"反光"的基准
    double max_v = 0.0;
    for (const auto& b : bars) max_v = std::max(max_v, b.mean_v);
    
    std::vector<Armor> armors;
    std::vector<bool> used(bars.size(), false);

    for(size_t i=0;i<bars.size();++i){
        if(used[i]) continue;
        for(size_t j=i+1;j<bars.size();++j){
            if(used[j]) continue;
            // const auto& left_bar = bars[i];
            // const auto& right_bar = bars[j];
            const bool i_is_left = (bars[i].center.x <= bars[j].center.x);
            const auto& left_bar = i_is_left ? bars[i] : bars[j];
            const auto& right_bar = i_is_left ? bars[j] : bars[i];

            double y_diff = std::abs(left_bar.center.y - right_bar.center.y);
            double x_diff = std::abs(left_bar.center.x - right_bar.center.x);
            double max_len = std::max(left_bar.length, right_bar.length);
            double max_w = std::max(left_bar.width, right_bar.width);

            double angle_diff = left_bar.angle - right_bar.angle;
            //灯条没有头尾，转 180° 看起来一样；直接相减会在 ±90 边界跳变
            //+89° 和 -89° 其实只差 2°，直接相减却得到 178°
            while (angle_diff >=  90.0) angle_diff -= 180.0;
            while (angle_diff <  -90.0) angle_diff += 180.0;

            if(y_diff > std::max(0.25 * max_len, 3.0) || std::abs(angle_diff) > 6.0) continue;
            
            if(std::abs(left_bar.length - right_bar.length) > std::max(0.20 * max_len, 4.0)) continue;//长度一致性
            if(std::abs(left_bar.width  - right_bar.width ) > std::max(0.22 * max_w, 2.0))   continue;//宽度一致性

            //   如果还有反光漏进来，把这个 0.93 调大（更严）；如果真板被误杀，调小
            if(max_v > 1.0 && std::min(left_bar.mean_v, right_bar.mean_v) < 0.93 * max_v) continue;
            
            //消除距离错觉
            double ratio = x_diff / max_len;

            //用于调试：检查灯条的配对情况
            // ROS_INFO_THROTTLE(1.0,"pair: ratio=%.2f  x_diff=%.0f  len=%.0f/%.0f  ang=%.1f/%.1f  ydiff=%.0f  V=%.0f/%.0f",
            //     ratio, x_diff,
            //     left_bar.length, right_bar.length,
            //     left_bar.angle, right_bar.angle,
            //     y_diff , left_bar.mean_v, right_bar.mean_v);

            if(ratio > 6.0 || ratio < 1.0) continue;
            //判断大小装甲板
            std::string bar_size = (ratio > 3.2) ? "Big" : "Small";

            int left_x = static_cast<int>(std::min(left_bar.center.x , right_bar.center.x));//左灯条x坐标
            int right_x = static_cast<int>(std::max(left_bar.center.x , right_bar.center.x));//右灯条
            double avg_y = (left_bar.center.y + right_bar.center.y) / 2.0;
            //找框的位置
            int min_x = static_cast<int>(left_x - max_len / 5);//稍微扩大一点
            int max_x = static_cast<int>(right_x + max_len / 5);
            int min_y = static_cast<int>(avg_y - max_len / 2);//下高度
            int max_y = static_cast<int>(avg_y + max_len / 2);//上高度
            
            Armor armor;
            //cv::Rect(左上角x，左上角y，宽，高)
            armor.bounding_box = cv::Rect(min_x,min_y,max_x - min_x,max_y - min_y);
            armor.type = bar_size;
            armor.color = color;
            //计算x轴和y轴的单位向量
            cv::Point2f ex = (right_bar.center - left_bar.center) /
                             static_cast<float>(cv::norm(right_bar.center - left_bar.center));
            cv::Point2f ey(-ex.y, ex.x);//向下

            // 高度用两根灯条长度的【均值】，不用 max：
            //   max 会被"其中一根粘上反光/白边、被量长了"带偏，
            //   而四角点的高度一旦虚高，PnP 就会以为板子在偏航（yaw 凭空出现），
            //   坐标轴整体歪掉、z 也会偏小。
            const double mean_len = 0.5 * (left_bar.length + right_bar.length);

            float half_len = static_cast<float>(mean_len * 0.5);
            armor.corners = {//左上 右上 右下 左下
                left_bar.center  - ey * half_len - ex * static_cast<float>(left_bar.width  * 0.5),
                right_bar.center - ey * half_len + ex * static_cast<float>(right_bar.width * 0.5),
                right_bar.center + ey * half_len + ex * static_cast<float>(right_bar.width * 0.5),
                left_bar.center  + ey * half_len - ex * static_cast<float>(left_bar.width  * 0.5) 
            };
            //打包合格的装甲板
            armors.push_back(armor);
            used[i] = true;
            used[j] = true;
            break;
        }
    }
    return armors;
}
//画框
void ArmorDetector::drawArmors(cv::Mat& image,const std::vector<Armor>& armors)
{
     for(size_t i=0;i<armors.size();++i){
        const auto& armor = armors[i];
        // OpenCV 是 BGR 顺序：蓝色 (255,0,0)，红色 (0,0,255)
       cv::Scalar draw_color = (armor.color == "Blue") ? cv::Scalar(255,0,0)
                                                       : cv::Scalar(0,0,255);
       cv::rectangle(image,armor.bounding_box,draw_color,2);//画框，线宽为2
       // 标记文字：#序号 大小 颜色
       std::string label = "#" + std::to_string(i) + " " + armor.type + " " + armor.color;
       int baseline = 0;
       cv::Size text_size = cv::getTextSize(label,
            cv::FONT_HERSHEY_SIMPLEX,0.5,1,&baseline);
       int ty = armor.bounding_box.y - 4;   // 默认画在框上方
       // 框贴到图像顶部放不下时，改画到框下方
       if (ty < text_size.height) {
           ty = armor.bounding_box.y + armor.bounding_box.height + text_size.height + 4;
       }
       cv::putText(image,label,cv::Point(armor.bounding_box.x,ty),
                   cv::FONT_HERSHEY_SIMPLEX,0.5,draw_color,1);
    }
}
//中控台
std::vector<Armor> ArmorDetector::detect(const cv::Mat& bgr_image)
{
    //将bgr图像转换成hsv
    cv::Mat hsv;
    cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);

    //蓝色装甲板
    cv::Mat mask_blue = getColorMask(hsv,true);
    auto blue_bars = findLightBars(mask_blue,hsv);
    auto blue_armors = matchArmors(blue_bars,"Blue");
    //红色装甲板
    cv::Mat mask_red = getColorMask(hsv,false);
    auto red_bars = findLightBars(mask_red,hsv);
    auto red_armors = matchArmors(red_bars,"Red");

    std::vector<Armor> all_armors = blue_armors;
    all_armors.insert(all_armors.end(),red_armors.begin(),red_armors.end());

    // 用于调试：每秒一行汇总：看清楚"灯条检出来了几根、最后配成了几块"，
    // ROS_INFO_THROTTLE(1.0,"detect: blue %zu->%zu bars->armors | red %zu->%zu bars->armors | total=%zu",
    //     blue_bars.size(), blue_armors.size(),
    //     red_bars.size(),  red_armors.size(), all_armors.size());
    
    return all_armors;
}