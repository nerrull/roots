#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "rtmdet_onnxruntime.h"
#include "rtmpose_utils.h"

class RTMPoseOnnxruntime {
public:
    RTMPoseOnnxruntime() = delete;
    explicit RTMPoseOnnxruntime(const std::string& onnx_model_path);

    std::vector<PosePoint> Inference(const cv::Mat& input_mat, const DetectBox& box);

private:
    std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat& img,
                                                      const DetectBox& box);
    void PrintModelInfo(Ort::Session& session);

    Ort::Env     m_env;
    Ort::Session m_session;
};
