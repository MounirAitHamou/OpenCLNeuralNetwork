#pragma once

#include "clnn/device.hpp"
#include "opencl_api.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace clnn::opencl {

class Buffer final {
  public:
    Buffer(Device device, std::size_t elements, const float* initial_data = nullptr);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] api::MemoryHandle handle() const noexcept;
    [[nodiscard]] std::vector<float> read() const;
    void write(const std::vector<float>& values);

  private:
    Device device_;
    std::size_t elements_;
    api::MemoryHandle handle_ = nullptr;
};

[[nodiscard]] std::shared_ptr<Buffer> clone(const Buffer& input);
void accumulate(Buffer& destination, const Buffer& contribution);

enum class BinaryOperation : api::UInt { add, subtract, multiply, divide };
enum class PoolingOperation : api::UInt { maximum, average };
enum class UnaryOperation : api::UInt {
    negate,
    power,
    exponential,
    logarithm,
    relu,
    sigmoid,
    hyperbolic_tangent
};

[[nodiscard]] bool available() noexcept;
[[nodiscard]] std::string device_name(Device device);
void synchronize(Device device);
void set_profiling(Device device, bool enabled);
[[nodiscard]] std::vector<KernelProfile> profile(Device device, bool reset);
[[nodiscard]] OpenCLRuntimeStatistics runtime_statistics(Device device);
void clear_memory_pool(Device device);
[[nodiscard]] std::shared_ptr<Buffer> binary(BinaryOperation operation, const Buffer& lhs,
                                             const Buffer& rhs,
                                             const std::vector<std::size_t>& lhs_indices,
                                             const std::vector<std::size_t>& rhs_indices);
[[nodiscard]] std::shared_ptr<Buffer> unary(UnaryOperation operation, const Buffer& input,
                                            float argument = 0.0F);
[[nodiscard]] std::shared_ptr<Buffer> matrix_multiply(const Buffer& lhs, const Buffer& rhs,
                                                      std::size_t rows, std::size_t inner,
                                                      std::size_t columns);
[[nodiscard]] std::shared_ptr<Buffer> reduce_sum(const Buffer& input);
[[nodiscard]] std::shared_ptr<Buffer> reduce_axis_sum(const Buffer& input, std::size_t outer,
                                                      std::size_t axis, std::size_t inner);
[[nodiscard]] std::shared_ptr<Buffer> axis_sum_gradient(const Buffer& upstream, std::size_t outer,
                                                        std::size_t axis, std::size_t inner);
[[nodiscard]] std::shared_ptr<Buffer> transpose(const Buffer& input, std::size_t rows,
                                                std::size_t columns);
[[nodiscard]] std::shared_ptr<Buffer> softmax(const Buffer& input, std::size_t outer,
                                              std::size_t axis, std::size_t inner);
[[nodiscard]] std::shared_ptr<Buffer>
convolution_2d(const Buffer& input, const Buffer& weight, const Buffer* bias, std::size_t batch,
               std::size_t input_channels, std::size_t input_height, std::size_t input_width,
               std::size_t output_channels, std::size_t kernel_height, std::size_t kernel_width,
               std::size_t output_height, std::size_t output_width, std::size_t stride_y,
               std::size_t stride_x, std::size_t padding_y, std::size_t padding_x);
[[nodiscard]] std::shared_ptr<Buffer>
pool_2d(PoolingOperation operation, const Buffer& input, std::size_t batch, std::size_t channels,
        std::size_t input_height, std::size_t input_width, std::size_t kernel_height,
        std::size_t kernel_width, std::size_t output_height, std::size_t output_width,
        std::size_t stride_y, std::size_t stride_x, std::size_t padding_y, std::size_t padding_x);
[[nodiscard]] std::shared_ptr<Buffer>
pool_2d_gradient(PoolingOperation operation, const Buffer& input, const Buffer& upstream,
                 std::size_t batch, std::size_t channels, std::size_t input_height,
                 std::size_t input_width, std::size_t kernel_height, std::size_t kernel_width,
                 std::size_t output_height, std::size_t output_width, std::size_t stride_y,
                 std::size_t stride_x, std::size_t padding_y, std::size_t padding_x);
[[nodiscard]] std::shared_ptr<Buffer>
binary_gradient(BinaryOperation operation, bool with_respect_to_lhs, const Buffer& lhs,
                const Buffer& rhs, const std::vector<std::size_t>& lhs_indices,
                const std::vector<std::size_t>& rhs_indices, const Buffer& upstream,
                std::size_t parent_size);
[[nodiscard]] std::shared_ptr<Buffer> unary_gradient(UnaryOperation operation, const Buffer& input,
                                                     const Buffer& upstream, float argument = 0.0F);
[[nodiscard]] std::shared_ptr<Buffer> sum_gradient(const Buffer& upstream, std::size_t count);
[[nodiscard]] std::shared_ptr<Buffer> matmul_lhs_gradient(const Buffer& rhs, const Buffer& upstream,
                                                          std::size_t rows, std::size_t inner,
                                                          std::size_t columns);
[[nodiscard]] std::shared_ptr<Buffer> matmul_rhs_gradient(const Buffer& lhs, const Buffer& upstream,
                                                          std::size_t rows, std::size_t inner,
                                                          std::size_t columns);
[[nodiscard]] std::shared_ptr<Buffer> softmax_gradient(const Buffer& output, const Buffer& upstream,
                                                       std::size_t outer, std::size_t axis,
                                                       std::size_t inner);
struct ConvolutionGradients final {
    std::shared_ptr<Buffer> input;
    std::shared_ptr<Buffer> weight;
    std::shared_ptr<Buffer> bias;
};
struct BinaryLossGradients final {
    std::shared_ptr<Buffer> prediction;
    std::shared_ptr<Buffer> target;
};
[[nodiscard]] std::shared_ptr<Buffer> binary_cross_entropy_loss(const Buffer& prediction,
                                                                const Buffer& target, float epsilon,
                                                                bool with_logits);
[[nodiscard]] BinaryLossGradients
binary_cross_entropy_gradients(const Buffer& prediction, const Buffer& target,
                               const Buffer& upstream, float epsilon, bool with_logits,
                               bool prediction_required, bool target_required);
[[nodiscard]] std::shared_ptr<Buffer> cross_entropy_loss(const Buffer& logits,
                                                         const std::vector<std::size_t>& labels,
                                                         std::size_t batch, std::size_t classes);
[[nodiscard]] std::shared_ptr<Buffer>
cross_entropy_gradient(const Buffer& logits, const std::vector<std::size_t>& labels,
                       const Buffer& upstream, std::size_t batch, std::size_t classes);
void sgd_update(Buffer& parameter, const Buffer& gradient, Buffer& velocity, float learning_rate,
                float momentum, float weight_decay);
void adam_update(Buffer& parameter, const Buffer& gradient, Buffer& first_moment,
                 Buffer& second_moment, float learning_rate, float beta1, float beta2,
                 float epsilon, float weight_decay, float first_correction, float second_correction,
                 bool decoupled_weight_decay);
[[nodiscard]] ConvolutionGradients
convolution_2d_gradients(const Buffer& input, const Buffer& weight, const Buffer& upstream,
                         bool input_required, bool weight_required, bool bias_required,
                         std::size_t batch, std::size_t input_channels, std::size_t input_height,
                         std::size_t input_width, std::size_t output_channels,
                         std::size_t kernel_height, std::size_t kernel_width,
                         std::size_t output_height, std::size_t output_width, std::size_t stride_y,
                         std::size_t stride_x, std::size_t padding_y, std::size_t padding_x);

} // namespace clnn::opencl
