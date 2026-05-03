#include "rtmdet_onnxruntime.h"
#include "coreml_provider_factory.h"

#include <algorithm>
#include <iostream>
#include <thread>

RTMDetOnnxruntime::RTMDetOnnxruntime(const std::string& onnx_model_path)
    : m_env(ORT_LOGGING_LEVEL_ERROR, "rtmdet"), m_session(nullptr)
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

DetectBox RTMDetOnnxruntime::Inference(const cv::Mat& input_mat)
{
    // RTMDet-nano has a fixed 320×320 input.
    // Resize, then scale output boxes back to original image space.
    constexpr int MODEL_W = 320, MODEL_H = 320;
    float sx = (float)input_mat.cols / MODEL_W;
    float sy = (float)input_mat.rows / MODEL_H;

    cv::Mat resized, rgb;
    cv::resize(input_mat, resized, {MODEL_W, MODEL_H}, 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    constexpr int C = 3;
    std::vector<float> buf(C * MODEL_H * MODEL_W);
    for (int h = 0; h < MODEL_H; h++)
        for (int w = 0; w < MODEL_W; w++)
            for (int c = 0; c < C; c++)
                buf[c * MODEL_H * MODEL_W + h * MODEL_W + w] =
                    (rgb.ptr<uchar>(h)[w * 3 + c] - IMAGE_MEAN[c]) / IMAGE_STD[c];

    std::array<int64_t, 4> shape{1, C, MODEL_H, MODEL_W};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value in_t = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(),
                                                       shape.data(), shape.size());

    const char* in_names[]  = {"input"};
    const char* out_names[] = {"dets", "labels"};
    auto outs = m_session.Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1,
                               out_names, 2);

    auto det_dims = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_dets  = (int)det_dims[1];
    int stride    = (int)det_dims[2];
    float* dets   = outs[0].GetTensorMutableData<float>();
    int*   labels = outs[1].GetTensorMutableData<int>();

    std::vector<DetectBox> boxes;
    for (int i = 0; i < num_dets; i++) {
        if (labels[i] != 0) continue;
        DetectBox b;
        b.left   = (int)(dets[i * stride + 0] * sx);
        b.top    = (int)(dets[i * stride + 1] * sy);
        b.right  = (int)(dets[i * stride + 2] * sx);
        b.bottom = (int)(dets[i * stride + 3] * sy);
        b.score  = dets[i * stride + 4];
        b.label  = labels[i];
        boxes.push_back(b);
    }
    std::sort(boxes.begin(), boxes.end(), BoxCompare);
    return boxes.empty() ? DetectBox{} : boxes[0];
}

void RTMDetOnnxruntime::PrintModelInfo(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t ni = session.GetInputCount(), no = session.GetOutputCount();
    std::cout << "[RTMDet] inputs=" << ni << " outputs=" << no << "\n";
    for (size_t i = 0; i < ni; i++)
        std::cout << "  in[" << i << "] " << session.GetInputNameAllocated(i, alloc) << "\n";
    for (size_t i = 0; i < no; i++)
        std::cout << "  out[" << i << "] " << session.GetOutputNameAllocated(i, alloc) << "\n";
}
