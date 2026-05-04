#include "rtmpose_onnxruntime.h"
#include "coreml_provider_factory.h"

#include <algorithm>
#include <iostream>
#include <thread>

RTMPoseOnnxruntime::RTMPoseOnnxruntime(const std::string& onnx_model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "rtmpose"), m_session(nullptr)
{
    int n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(n_threads);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    opts.SetLogSeverityLevel(4);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(opts, 0));

    m_session = Ort::Session(m_env, onnx_model_path.c_str(), opts);
    PrintModelInfo(m_session);
}

std::vector<PosePoint> RTMPoseOnnxruntime::Inference(const cv::Mat& input_mat,
                                                       const DetectBox& box)
{
    std::vector<PosePoint> result;
    if (!box.IsValid()) return result;

    auto [crop, inv_affine] = CropImageByDetectBox(input_mat, box);

    cv::Mat rgb;
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);

    int H = rgb.rows, W = rgb.cols, C = rgb.channels();
    std::vector<float> buf(C * H * W);
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++)
                buf[c * H * W + h * W + w] =
                    (rgb.ptr<uchar>(h)[w * 3 + c] - IMAGE_MEAN[c]) / IMAGE_STD[c];

    std::array<int64_t, 4> shape{1, C, H, W};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value in_t = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(),
                                                       shape.data(), shape.size());

    const char* in_names[]  = {"input"};
    const char* out_names[] = {"simcc_x", "simcc_y"};
    auto outs = m_session.Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1,
                               out_names, 2);

    auto x_dims  = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto y_dims  = outs[1].GetTensorTypeAndShapeInfo().GetShape();
    int  joints  = (int)x_dims[1];
    int  ext_w   = (int)x_dims[2];
    int  ext_h   = (int)y_dims[2];
    float* sx    = outs[0].GetTensorMutableData<float>();
    float* sy    = outs[1].GetTensorMutableData<float>();

    for (int i = 0; i < joints; i++) {
        auto* rx = sx + i * ext_w;
        auto* ry = sy + i * ext_h;
        auto  xi = std::max_element(rx, rx + ext_w);
        auto  yi = std::max_element(ry, ry + ext_h);

        PosePoint p;
        p.x     = (int)(std::distance(rx, xi) / 2);
        p.y     = (int)(std::distance(ry, yi) / 2);
        p.score = (*xi + *yi) * 0.5f;
        result.push_back(p);
    }

    // map back to original image coordinates
    for (auto& p : result) {
        cv::Mat pt = cv::Mat::ones(3, 1, CV_64FC1);
        pt.at<double>(0) = p.x;
        pt.at<double>(1) = p.y;
        cv::Mat res = inv_affine * pt;
        p.x = (int)res.at<double>(0);
        p.y = (int)res.at<double>(1);
    }

    return result;
}

std::pair<cv::Mat, cv::Mat>
RTMPoseOnnxruntime::CropImageByDetectBox(const cv::Mat& img, const DetectBox& box)
{
    int bw = box.right  - box.left;
    int bh = box.bottom - box.top;
    int bcx = box.left + bw / 2;
    int bcy = box.top  + bh / 2;

    constexpr float aspect = 192.f / 256.f;
    if (bw > aspect * bh)      bh = (int)(bw / aspect);
    else if (bw < aspect * bh) bw = (int)(bh * aspect);

    float sw = bw * 1.2f, sh = bh * 1.2f;

    cv::Mat fwd = GetAffineTransform((float)bcx, (float)bcy, sw, sh, 192, 256);
    cv::Mat inv = GetAffineTransform((float)bcx, (float)bcy, sw, sh, 192, 256, true);

    cv::Mat crop;
    cv::warpAffine(img, crop, fwd, cv::Size(192, 256), cv::INTER_LINEAR);
    return {crop, inv};
}

void RTMPoseOnnxruntime::PrintModelInfo(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t ni = session.GetInputCount(), no = session.GetOutputCount();
    std::cout << "[RTMPose] inputs=" << ni << " outputs=" << no << "\n";
    for (size_t i = 0; i < ni; i++)
        std::cout << "  in[" << i << "] " << session.GetInputNameAllocated(i, alloc) << "\n";
    for (size_t i = 0; i < no; i++)
        std::cout << "  out[" << i << "] " << session.GetOutputNameAllocated(i, alloc) << "\n";
}
