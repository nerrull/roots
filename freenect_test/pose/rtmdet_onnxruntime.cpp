#include "rtmdet_onnxruntime.h"
#include "coreml_provider_factory.h"

#include <algorithm>
#include <cmath>
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

std::vector<DetectBox> RTMDetOnnxruntime::Inference(const cv::Mat& input_mat)
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
    // MMDeploy RTMDet exports labels as int64 — reading as int32 misaligns
    // the array, letting non-person detections leak through the class filter.
    const int64_t* labels = outs[1].GetTensorData<int64_t>();

    constexpr float SCORE_THRESH = 0.5f;
    constexpr float NMS_IOU_THRESH = 0.45f;
    constexpr float MIN_BOX_PX = 4.f;
    const float MODEL_W_F = (float)MODEL_W, MODEL_H_F = (float)MODEL_H;

    std::vector<DetectBox> boxes;
    for (int i = 0; i < num_dets; i++) {
        if (labels[i] != 0) continue;
        float score = dets[i * stride + 4];
        if (score < SCORE_THRESH) continue;

        float x1 = dets[i * stride + 0];
        float y1 = dets[i * stride + 1];
        float x2 = dets[i * stride + 2];
        float y2 = dets[i * stride + 3];

        // Drop garbage padding slots: non-finite, degenerate, or far OOB.
        if (!std::isfinite(x1) || !std::isfinite(y1) ||
            !std::isfinite(x2) || !std::isfinite(y2)) continue;
        if (x2 - x1 < MIN_BOX_PX || y2 - y1 < MIN_BOX_PX) continue;
        if (x2 < 0.f || y2 < 0.f || x1 > MODEL_W_F || y1 > MODEL_H_F) continue;

        // Clip to model space before scaling back to source-image space.
        x1 = std::max(0.f, std::min(MODEL_W_F, x1));
        y1 = std::max(0.f, std::min(MODEL_H_F, y1));
        x2 = std::max(0.f, std::min(MODEL_W_F, x2));
        y2 = std::max(0.f, std::min(MODEL_H_F, y2));

        DetectBox b;
        b.left   = (int)(x1 * sx);
        b.top    = (int)(y1 * sy);
        b.right  = (int)(x2 * sx);
        b.bottom = (int)(y2 * sy);
        b.score  = score;
        b.label  = (int)labels[i];
        boxes.push_back(b);
    }
    std::sort(boxes.begin(), boxes.end(), BoxCompare);

    // NMS: suppress boxes with high IoU overlap against higher-scoring ones
    std::vector<DetectBox> kept;
    std::vector<bool> suppressed(boxes.size(), false);
    for (int i = 0; i < (int)boxes.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(boxes[i]);
        for (int j = i + 1; j < (int)boxes.size(); j++) {
            if (suppressed[j]) continue;
            int ix = std::max(boxes[i].left,  boxes[j].left);
            int iy = std::max(boxes[i].top,   boxes[j].top);
            int ax = std::min(boxes[i].right,  boxes[j].right);
            int ay = std::min(boxes[i].bottom, boxes[j].bottom);
            if (ax > ix && ay > iy) {
                float inter = (float)(ax-ix) * (float)(ay-iy);
                float ua    = (float)(boxes[i].right-boxes[i].left) * (float)(boxes[i].bottom-boxes[i].top);
                float ub    = (float)(boxes[j].right-boxes[j].left) * (float)(boxes[j].bottom-boxes[j].top);
                if (inter / (ua + ub - inter) > NMS_IOU_THRESH)
                    suppressed[j] = true;
            }
        }
    }
    return kept;
}

void RTMDetOnnxruntime::PrintModelInfo(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t ni = session.GetInputCount(), no = session.GetOutputCount();
    std::cout << "[RTMDet] inputs=" << ni << " outputs=" << no << "\n";
    for (size_t i = 0; i < ni; i++) {
        auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto dims = info.GetShape();
        std::cout << "  in[" << i << "] " << session.GetInputNameAllocated(i, alloc) << " shape=[";
        for (size_t j = 0; j < dims.size(); j++) std::cout << (j?",":"") << dims[j];
        std::cout << "]\n";
    }
    for (size_t i = 0; i < no; i++) {
        auto info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto dims = info.GetShape();
        std::cout << "  out[" << i << "] " << session.GetOutputNameAllocated(i, alloc)
                  << " dtype=" << info.GetElementType() << " shape=[";
        for (size_t j = 0; j < dims.size(); j++) std::cout << (j?",":"") << dims[j];
        std::cout << "]\n";
    }

    // Validate that the model matches the assumptions baked into Inference().
    auto in_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape.size() != 4 || in_shape[2] != 320 || in_shape[3] != 320)
        fprintf(stderr, "[RTMDet] WARNING: expected input shape 1x3x320x320 but got something different\n");
    auto labels_type = session.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetElementType();
    if (labels_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        fprintf(stderr, "[RTMDet] WARNING: labels output is not int64 — Inference() will misread it\n");
}
