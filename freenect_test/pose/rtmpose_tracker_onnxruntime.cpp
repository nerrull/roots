#include "rtmpose_tracker_onnxruntime.h"

RTMPoseTrackerOnnxruntime::RTMPoseTrackerOnnxruntime(const std::string& det_path,
                                                      const std::string& pose_path,
                                                      int detect_interval)
    : m_detect_interval(detect_interval)
{
    m_det  = std::make_unique<RTMDetOnnxruntime>(det_path);
    m_pose = std::make_unique<RTMPoseOnnxruntime>(pose_path);
}

std::vector<std::pair<DetectBox, std::vector<PosePoint>>>
RTMPoseTrackerOnnxruntime::Inference(const cv::Mat& frame)
{
    if (m_frame_num % m_detect_interval == 0)
        m_boxes = m_det->Inference(frame);

    std::vector<std::pair<DetectBox, std::vector<PosePoint>>> results;
    results.reserve(m_boxes.size());
    for (auto& box : m_boxes) {
        if (!box.IsValid()) continue;
        results.push_back({box, m_pose->Inference(frame, box)});
    }
    m_frame_num++;
    return results;
}
