#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "clnn/device.hpp"

namespace clnn {

using Shape = std::vector<std::size_t>;

namespace detail {
struct TensorImpl;
}
namespace opencl {
class Buffer;
}

class Tensor final {
  public:
    Tensor() = default;
    Tensor(std::vector<float> data, Shape shape, bool requires_grad = false, std::string name = {},
           Device device = Device::cpu());

    [[nodiscard]] static Tensor scalar(float value, bool requires_grad = false,
                                       std::string name = {}, Device device = Device::cpu());
    [[nodiscard]] static Tensor zeros(const Shape& shape, bool requires_grad = false,
                                      std::string name = {}, Device device = Device::cpu());
    [[nodiscard]] static Tensor ones(const Shape& shape, bool requires_grad = false,
                                     std::string name = {}, Device device = Device::cpu());
    [[nodiscard]] static Tensor randn(const Shape& shape, std::mt19937& generator,
                                      float mean = 0.0F, float standard_deviation = 1.0F,
                                      bool requires_grad = false, std::string name = {},
                                      Device device = Device::cpu());

    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] std::size_t ndim() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool is_scalar() const;
    [[nodiscard]] float item() const;
    [[nodiscard]] const std::vector<float>& data() const;
    [[nodiscard]] const std::vector<float>& grad() const;
    [[nodiscard]] bool has_grad() const;
    [[nodiscard]] bool requires_grad() const;
    [[nodiscard]] bool is_leaf() const;
    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] Device device() const;
    [[nodiscard]] Tensor to(const Device& device) const;

    void backward(std::optional<Tensor> gradient = std::nullopt, bool retain_graph = false);
    void zero_grad();
    void set_data(std::vector<float> data);
    [[nodiscard]] Tensor detach() const;

  private:
    explicit Tensor(std::shared_ptr<detail::TensorImpl> implementation);
    [[nodiscard]] const std::shared_ptr<detail::TensorImpl>& impl() const;

    std::shared_ptr<detail::TensorImpl> implementation_;

    friend Tensor detail_make_result(std::vector<float>, Shape, std::vector<Tensor>,
                                     std::function<void(const Tensor&)>, std::string);
    friend void detail_accumulate(const Tensor&, const std::vector<float>&);
    friend void detail_accumulate(const Tensor&, const Tensor&);
    friend std::shared_ptr<opencl::Buffer> detail_opencl_buffer(const Tensor&);
    friend std::shared_ptr<opencl::Buffer> detail_opencl_gradient_buffer(const Tensor&);
    friend Tensor detail_make_opencl_result(std::shared_ptr<opencl::Buffer>, Shape,
                                            std::vector<Tensor>, std::function<void(const Tensor&)>,
                                            std::string, Device);
    friend void detail_mark_opencl_modified(Tensor&);
    friend Tensor detail_clone_leaf(const Tensor&, bool, std::string);
    friend void detail_backward_to(Tensor, const Tensor&, std::optional<Tensor>);
    friend Tensor detail_gradient_tensor(const Tensor&);
};

class NoGradGuard final {
  public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

  private:
    bool previous_;
};

namespace detail {
class GradRecordingGuard final {
  public:
    GradRecordingGuard();
    ~GradRecordingGuard();
    GradRecordingGuard(const GradRecordingGuard&) = delete;
    GradRecordingGuard& operator=(const GradRecordingGuard&) = delete;

  private:
    bool previous_;
};
} // namespace detail

[[nodiscard]] bool grad_enabled() noexcept;
[[nodiscard]] std::size_t numel(const Shape& shape);
[[nodiscard]] std::string shape_string(const Shape& shape);

// Internal operation hook. Public only so operations can live in separate translation units.
Tensor detail_make_result(std::vector<float> data, Shape shape, std::vector<Tensor> parents,
                          std::function<void(const Tensor&)> backward, std::string operation);
void detail_accumulate(const Tensor& tensor, const std::vector<float>& gradient);
void detail_accumulate(const Tensor& tensor, const Tensor& gradient);
std::shared_ptr<opencl::Buffer> detail_opencl_buffer(const Tensor& tensor);
std::shared_ptr<opencl::Buffer> detail_opencl_gradient_buffer(const Tensor& tensor);
Tensor detail_make_opencl_result(std::shared_ptr<opencl::Buffer> buffer, Shape shape,
                                 std::vector<Tensor> parents,
                                 std::function<void(const Tensor&)> backward, std::string operation,
                                 Device device);
void detail_mark_opencl_modified(Tensor& tensor);
Tensor detail_clone_leaf(const Tensor& tensor, bool requires_grad, std::string name = {});
void detail_backward_to(Tensor output, const Tensor& input,
                        std::optional<Tensor> gradient = std::nullopt);
[[nodiscard]] Tensor detail_gradient_tensor(const Tensor& tensor);

} // namespace clnn
