#include "rtmpose_tracker_onnxruntime.h"

RTMPoseTrackerOnnxruntime::RTMPoseTrackerOnnxruntime(const std::string& det_path,
                                                      const std::string& pose_path,
                                                      int detect_interval)
    : m_detect_interval(detect_interval)
{
    m_det  = std::make_unique<RTMDetOnnxruntime>(det_path);
    m_pose = std::make_unique<RTMPoseOnnxruntime>(pose_path);
}

std::pair<DetectBox, std::vector<PosePoint>>
RTMPoseTrackerOnnxruntime::Inference(const cv::Mat& frame)
{
    if (m_frame_num % m_detect_interval == 0)
        m_box = m_det->Inference(frame);

    auto kps = m_pose->Inference(frame, m_box);
    m_frame_num++;
    return {m_box, kps};
}
