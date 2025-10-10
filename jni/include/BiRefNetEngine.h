#ifndef LITEMIND_BIREFNET_ENGINE_H_
#define LITEMIND_BIREFNET_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

namespace MNN {
class Interpreter;
class Session;
class Tensor;
struct ScheduleConfig;
}  // namespace MNN

namespace litemind {

struct Size {
    int width = 0;
    int height = 0;
};

class BiRefNetEngine {
public:
    BiRefNetEngine();
    ~BiRefNetEngine();

    BiRefNetEngine(const BiRefNetEngine&) = delete;
    BiRefNetEngine& operator=(const BiRefNetEngine&) = delete;

    // 加载 MNN 模型并初始化 Session。threads 用于控制 CPU 线程数。
    void Initialize(const std::string& model_path, int threads = 4);

    // 对输入位图（RGBA8888）执行推理，返回与原图同尺寸的掩码（0-255）。
    std::vector<uint8_t> Run(const uint8_t* pixels_rgba,
                             const Size& input_size,
                             const Size& output_size);

    Size model_input_size() const { return model_input_size_; }

private:
    void Release();
    void EnsureSession();
    void Preprocess(const uint8_t* pixels_rgba,
                    const Size& src_size,
                    std::vector<float>& nchw_buffer) const;
    std::vector<uint8_t> Postprocess(MNN::Tensor* output_tensor,
                                     const Size& dst_size) const;

    std::unique_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    Size model_input_size_{512, 512};  // BiRefNet 默认输入
    int threads_ = 4;
    std::string input_tensor_name_ = "input_image";
    std::string output_tensor_name_ = "output_image";
};

}  // namespace litemind

#endif  // LITEMIND_BIREFNET_ENGINE_H_
