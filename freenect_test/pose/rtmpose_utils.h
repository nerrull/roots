#pragma once

#include "opencv2/opencv.hpp"

const std::vector<float> IMAGE_MEAN{ 123.675, 116.28, 103.53 };
const std::vector<float> IMAGE_STD{ 58.395, 57.12, 57.375 };

struct DetectBox {
    int left, top, right, bottom;
    float score;
    int label;

    DetectBox() : left(-1), top(-1), right(-1), bottom(-1), score(-1.f), label(-1) {}

    bool IsValid() const {
        return left != -1 && top != -1 && right != -1 && bottom != -1
            && score != -1.f && label != -1;
    }
};

static bool BoxCompare(const DetectBox& a, const DetectBox& b) {
    return a.score > b.score;
}

struct PosePoint {
    int x, y;
    float score;
    PosePoint() : x(0), y(0), score(0.f) {}
};

static cv::Mat GetAffineTransform(float cx, float cy, float sw, float sh,
                                   int ow, int oh, bool inverse = false)
{
    cv::Point2f src[3], dst[3];
    src[0] = {cx, cy};
    src[1] = {cx, cy - sw * 0.5f};
    src[2] = {src[1].x - (src[0].y - src[1].y), src[1].y + (src[0].x - src[1].x)};

    float dcx = ow / 2.f, dcy = oh / 2.f;
    dst[0] = {dcx, dcy};
    dst[1] = {dcx, dcy - ow * 0.5f};
    dst[2] = {dst[1].x - (dst[0].y - dst[1].y), dst[1].y + (dst[0].x - dst[1].x)};

    return inverse ? cv::getAffineTransform(dst, src) : cv::getAffineTransform(src, dst);
}
