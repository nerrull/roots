#pragma once

#include <string>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "rtmpose_utils.h"

class RTMDetOnnxruntime {
public:
    RTMDetOnnxruntime() = delete;
    explicit RTMDetOnnxruntime(const std::string& onnx_model_path);

    std::vector<DetectBox> Inference(const cv::Mat& input_mat);

private:
    void PrintModelInfo(Ort::Session& session);

    Ort::Env     m_env;
    Ort::Session m_session;
};
