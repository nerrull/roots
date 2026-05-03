#pragma once

#include <memory>
#include <vector>
#include "rtmdet_onnxruntime.h"
#include "rtmpose_onnxruntime.h"

class RTMPoseTrackerOnnxruntime {
public:
    RTMPoseTrackerOnnxruntime() = delete;
    RTMPoseTrackerOnnxruntime(const std::string& det_path,
                               const std::string& pose_path,
                               int detect_interval = 10);

    std::pair<DetectBox, std::vector<PosePoint>> Inference(const cv::Mat& frame);

private:
    std::unique_ptr<RTMDetOnnxruntime>  m_det;
    std::unique_ptr<RTMPoseOnnxruntime> m_pose;
    DetectBox    m_box;
    unsigned int m_frame_num{0};
    int          m_detect_interval;
};
