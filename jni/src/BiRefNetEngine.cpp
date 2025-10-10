#include "BiRefNetEngine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#define LITEMIND_LOG_TAG "LiteMindBiRefNet"
#define LITEMIND_LOGI(...) __android_log_print(ANDROID_LOG_INFO, LITEMIND_LOG_TAG, __VA_ARGS__)
#define LITEMIND_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LITEMIND_LOG_TAG, __VA_ARGS__)
#else
#include <iostream>
#define LITEMIND_LOGI(...) (static_cast<void>(0))
#define LITEMIND_LOGE(...) (std::cerr << "[BiRefNetEngine] " << __VA_ARGS__ << std::endl)
#endif

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

inline float Sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

inline int ClampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

}  // namespace

namespace litemind {

BiRefNetEngine::BiRefNetEngine() = default;

BiRefNetEngine::~BiRefNetEngine() {
    Release();
}

void BiRefNetEngine::Release() {
    if (interpreter_ && session_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    interpreter_.reset();
}

void BiRefNetEngine::EnsureSession() {
    if (!interpreter_) {
        throw std::runtime_error("BiRefNetEngine not initialized. Call Initialize() first.");
    }
    if (!session_) {
        throw std::runtime_error("MNN session not ready.");
    }
}

void BiRefNetEngine::Initialize(const std::string& model_path, int threads) {
    Release();

    threads_ = std::max(1, threads);
    interpreter_.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!interpreter_) {
        throw std::runtime_error("Failed to create MNN interpreter. Check model path.");
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    config.numThread = threads_;

    MNN::BackendConfig backend_config;
    backend_config.power = MNN::BackendConfig::Power_High;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    backend_config.memory = MNN::BackendConfig::Memory_Normal;
    config.backendConfig = &backend_config;

    session_ = interpreter_->createSession(config);
    if (!session_) {
        throw std::runtime_error("Failed to create MNN session.");
    }

    // 读取模型输入输出名称（若名称不匹配，采用第一个 Tensor）
    const auto inputs = interpreter_->getSessionInputAll(session_);
    if (!inputs.empty()) {
        input_tensor_name_ = inputs.begin()->first;
        auto shape = inputs.begin()->second->shape();
        if (shape.size() == 4) {
            model_input_size_.height = shape[2];
            model_input_size_.width = shape[3];
        }
    }

    const auto outputs = interpreter_->getSessionOutputAll(session_);
    if (!outputs.empty()) {
        output_tensor_name_ = outputs.begin()->first;
    }

    LITEMIND_LOGI("BiRefNetEngine initialized. Model: %s, input: %s, output: %s",
                  model_path.c_str(),
                  input_tensor_name_.c_str(),
                  output_tensor_name_.c_str());
}

std::vector<uint8_t> BiRefNetEngine::Run(const uint8_t* pixels_rgba,
                                         const Size& input_size,
                                         const Size& output_size) {
    EnsureSession();

    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::invalid_argument("Invalid source size.");
    }
    if (output_size.width <= 0 || output_size.height <= 0) {
        throw std::invalid_argument("Invalid destination size.");
    }
    if (pixels_rgba == nullptr) {
        throw std::invalid_argument("pixels_rgba is null.");
    }

    const Size model_size = model_input_size_;
    std::vector<float> nchw_buffer(3 * model_size.width * model_size.height);
    Preprocess(pixels_rgba, input_size, nchw_buffer);

    const char* input_name =
        input_tensor_name_.empty() ? nullptr : input_tensor_name_.c_str();
    auto* input_tensor = interpreter_->getSessionInput(session_, input_name);
    if (!input_tensor) {
        throw std::runtime_error("Failed to fetch input tensor.");
    }

    std::vector<int> shape = {1, 3, model_size.height, model_size.width};
    interpreter_->resizeTensor(input_tensor, shape);
    interpreter_->resizeSession(session_);

    std::unique_ptr<MNN::Tensor> host_tensor(
        MNN::Tensor::create<float>(shape, nchw_buffer.data(), MNN::Tensor::CAFFE));
    input_tensor->copyFromHostTensor(host_tensor.get());

    interpreter_->runSession(session_);

    const char* output_name =
        output_tensor_name_.empty() ? nullptr : output_tensor_name_.c_str();
    auto* output_tensor = interpreter_->getSessionOutput(session_, output_name);
    if (!output_tensor) {
        throw std::runtime_error("Failed to fetch output tensor.");
    }

    auto dims = output_tensor->shape();
    if (dims.size() != 4) {
        throw std::runtime_error("Unexpected output tensor shape.");
    }

    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);

    return Postprocess(&output_host, output_size);
}

void BiRefNetEngine::Preprocess(const uint8_t* pixels_rgba,
                                const Size& src_size,
                                std::vector<float>& nchw_buffer) const {
    const int src_w = src_size.width;
    const int src_h = src_size.height;
    const int dst_w = model_input_size_.width;
    const int dst_h = model_input_size_.height;

    auto* dst_c0 = nchw_buffer.data();
    auto* dst_c1 = dst_c0 + dst_w * dst_h;
    auto* dst_c2 = dst_c1 + dst_w * dst_h;

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = ClampInt(static_cast<int>(std::floor(src_y)), 0, src_h - 1);
        const int y1 = ClampInt(y0 + 1, 0, src_h - 1);
        const float ly = src_y - static_cast<float>(y0);

        for (int x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = ClampInt(static_cast<int>(std::floor(src_x)), 0, src_w - 1);
            const int x1 = ClampInt(x0 + 1, 0, src_w - 1);
            const float lx = src_x - static_cast<float>(x0);

            const int idx00 = (y0 * src_w + x0) * 4;
            const int idx01 = (y0 * src_w + x1) * 4;
            const int idx10 = (y1 * src_w + x0) * 4;
            const int idx11 = (y1 * src_w + x1) * 4;

            const float w00 = (1.f - lx) * (1.f - ly);
            const float w01 = lx * (1.f - ly);
            const float w10 = (1.f - lx) * ly;
            const float w11 = lx * ly;

            const int dst_index = y * dst_w + x;

            for (int c = 0; c < 3; ++c) {
                const float v00 = static_cast<float>(pixels_rgba[idx00 + c]) / 255.f;
                const float v01 = static_cast<float>(pixels_rgba[idx01 + c]) / 255.f;
                const float v10 = static_cast<float>(pixels_rgba[idx10 + c]) / 255.f;
                const float v11 = static_cast<float>(pixels_rgba[idx11 + c]) / 255.f;

                const float value = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                const float normalized = (value - kMean[c]) / kStd[c];

                float* dst_ptr = (c == 0) ? dst_c2 : (c == 1 ? dst_c1 : dst_c0);
                // 注意：Android ARGB -> 通道顺序需要变换，BiRefNet 期望输入为 RGB
                dst_ptr[dst_index] = normalized;
            }
        }
    }
}

std::vector<uint8_t> BiRefNetEngine::Postprocess(MNN::Tensor* output_tensor,
                                                 const Size& dst_size) const {
    const auto& shape = output_tensor->shape();
    const int src_h = shape[2];
    const int src_w = shape[3];

    const float* src_data = output_tensor->host<float>();
    std::vector<uint8_t> mask(dst_size.width * dst_size.height);

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_size.width);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_size.height);

    for (int y = 0; y < dst_size.height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = ClampInt(static_cast<int>(std::floor(src_y)), 0, src_h - 1);
        const int y1 = ClampInt(y0 + 1, 0, src_h - 1);
        const float ly = src_y - static_cast<float>(y0);

        for (int x = 0; x < dst_size.width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = ClampInt(static_cast<int>(std::floor(src_x)), 0, src_w - 1);
            const int x1 = ClampInt(x0 + 1, 0, src_w - 1);
            const float lx = src_x - static_cast<float>(x0);

            const float w00 = (1.f - lx) * (1.f - ly);
            const float w01 = lx * (1.f - ly);
            const float w10 = (1.f - lx) * ly;
            const float w11 = lx * ly;

            const int idx00 = y0 * src_w + x0;
            const int idx01 = y0 * src_w + x1;
            const int idx10 = y1 * src_w + x0;
            const int idx11 = y1 * src_w + x1;

            const float logit = src_data[idx00] * w00 + src_data[idx01] * w01 +
                                src_data[idx10] * w10 + src_data[idx11] * w11;
            const uint8_t mask_val = static_cast<uint8_t>(std::round(Sigmoid(logit) * 255.f));
            mask[y * dst_size.width + x] = mask_val;
        }
    }

    return mask;
}

}  // namespace litemind
