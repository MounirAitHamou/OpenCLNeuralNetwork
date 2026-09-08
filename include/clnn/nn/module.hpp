#pragma once

#include "clnn/autograd/ops.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace clnn::nn {

namespace detail {
struct ModuleHookState;
}

enum class ModuleType : std::uint32_t {
    custom = 0,
    linear,
    convolution_2d,
    relu,
    sigmoid,
    tanh,
    flatten,
    leaky_relu,
    softmax,
    max_pool_2d,
    average_pool_2d,
    dropout,
    batch_normalization,
    layer_normalization,
    global_average_pool_2d,
    residual
};
enum class PaddingMode { valid, same };

struct NamedParameter final {
    std::string name;
    Tensor* tensor;
};

class Module;
using ForwardHook = std::function<void(Module&, const Tensor&, const Tensor&)>;

class ForwardHookHandle final {
  public:
    ForwardHookHandle() = default;
    ~ForwardHookHandle();
    ForwardHookHandle(const ForwardHookHandle&) = delete;
    ForwardHookHandle& operator=(const ForwardHookHandle&) = delete;
    ForwardHookHandle(ForwardHookHandle&& other) noexcept;
    ForwardHookHandle& operator=(ForwardHookHandle&& other) noexcept;

    void remove() noexcept;
    [[nodiscard]] bool active() const noexcept;

  private:
    ForwardHookHandle(std::weak_ptr<detail::ModuleHookState> state, std::uint64_t identifier);

    std::weak_ptr<detail::ModuleHookState> state_;
    std::uint64_t identifier_ = 0;

    friend class Module;
};

class Module {
  public:
    Module();
    virtual ~Module() = default;
    Module(const Module& other);
    Module& operator=(const Module& other);
    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;
    [[nodiscard]] virtual Tensor forward(const Tensor& input) = 0;
    [[nodiscard]] virtual std::vector<NamedParameter> named_parameters();
    [[nodiscard]] virtual std::vector<NamedParameter> named_buffers();
    [[nodiscard]] virtual ModuleType type() const noexcept;
    [[nodiscard]] std::vector<Tensor*> parameters();
    [[nodiscard]] Tensor operator()(const Tensor& input);
    virtual Module& to(const Device& device);
    virtual Module& train(bool training = true);
    Module& eval();
    [[nodiscard]] bool training() const noexcept;
    [[nodiscard]] ForwardHookHandle register_forward_hook(ForwardHook hook);

  protected:
    bool training_ = true;

  private:
    std::shared_ptr<detail::ModuleHookState> hook_state_;
};

class Linear final : public Module {
  public:
    Linear(std::size_t input_features, std::size_t output_features, std::mt19937& generator,
           bool use_bias = true, Device device = Device::cpu());

    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] Tensor& weight() noexcept;
    [[nodiscard]] Tensor& bias();
    [[nodiscard]] std::size_t input_features() const noexcept;
    [[nodiscard]] std::size_t output_features() const noexcept;
    [[nodiscard]] bool has_bias() const noexcept;
    [[nodiscard]] ModuleType type() const noexcept override;

  private:
    std::size_t input_features_;
    std::size_t output_features_;
    Tensor weight_;
    std::optional<Tensor> bias_;
};

class ReLU final : public Module {
  public:
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
};

class Sigmoid final : public Module {
  public:
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
};

class Tanh final : public Module {
  public:
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
};

class LeakyReLU final : public Module {
  public:
    explicit LeakyReLU(float negative_slope = 0.01F);
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] float negative_slope() const noexcept;

  private:
    float negative_slope_;
};

class Softmax final : public Module {
  public:
    explicit Softmax(std::size_t dimension = 1);
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] std::size_t dimension() const noexcept;

  private:
    std::size_t dimension_;
};

class Flatten final : public Module {
  public:
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
};

class MaxPool2d final : public Module {
  public:
    explicit MaxPool2d(std::array<std::size_t, 2> kernel_size,
                       std::array<std::size_t, 2> stride = {0, 0},
                       std::array<std::size_t, 2> padding = {0, 0});
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] std::array<std::size_t, 2> kernel_size() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> stride() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> padding() const noexcept;

  private:
    std::array<std::size_t, 2> kernel_size_;
    std::array<std::size_t, 2> stride_;
    std::array<std::size_t, 2> padding_;
};

class AvgPool2d final : public Module {
  public:
    explicit AvgPool2d(std::array<std::size_t, 2> kernel_size,
                       std::array<std::size_t, 2> stride = {0, 0},
                       std::array<std::size_t, 2> padding = {0, 0});
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] std::array<std::size_t, 2> kernel_size() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> stride() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> padding() const noexcept;

  private:
    std::array<std::size_t, 2> kernel_size_;
    std::array<std::size_t, 2> stride_;
    std::array<std::size_t, 2> padding_;
};

class Dropout final : public Module {
  public:
    explicit Dropout(float probability = 0.5F, std::uint32_t seed = 0);
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] float probability() const noexcept;

  private:
    float probability_;
    std::mt19937 generator_;
};

class BatchNorm final : public Module {
  public:
    explicit BatchNorm(std::size_t features, float epsilon = 1.0e-5F, float momentum = 0.1F,
                       bool affine = true, Device device = Device::cpu());
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] std::vector<NamedParameter> named_buffers() override;
    [[nodiscard]] ModuleType type() const noexcept override;
    Module& to(const Device& device) override;
    [[nodiscard]] std::size_t features() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] float momentum() const noexcept;
    [[nodiscard]] bool affine() const noexcept;
    [[nodiscard]] Tensor& weight();
    [[nodiscard]] Tensor& bias();
    [[nodiscard]] const Tensor& running_mean() const noexcept;
    [[nodiscard]] const Tensor& running_variance() const noexcept;

  private:
    std::size_t features_;
    float epsilon_;
    float momentum_;
    std::optional<Tensor> weight_;
    std::optional<Tensor> bias_;
    Tensor running_mean_;
    Tensor running_variance_;
};

class LayerNorm final : public Module {
  public:
    explicit LayerNorm(Shape normalized_shape, float epsilon = 1.0e-5F, bool affine = true,
                       Device device = Device::cpu());
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] ModuleType type() const noexcept override;
    [[nodiscard]] const Shape& normalized_shape() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] bool affine() const noexcept;
    [[nodiscard]] Tensor& weight();
    [[nodiscard]] Tensor& bias();

  private:
    Shape normalized_shape_;
    float epsilon_;
    std::optional<Tensor> weight_;
    std::optional<Tensor> bias_;
};

class GlobalAvgPool2d final : public Module {
  public:
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] ModuleType type() const noexcept override;
};

class Residual final : public Module {
  public:
    explicit Residual(std::unique_ptr<Module> module);
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] std::vector<NamedParameter> named_buffers() override;
    [[nodiscard]] ModuleType type() const noexcept override;
    Module& to(const Device& device) override;
    Module& train(bool training = true) override;
    [[nodiscard]] Module& module() noexcept;

  private:
    std::unique_ptr<Module> module_;
};

class Conv2d final : public Module {
  public:
    Conv2d(std::size_t input_channels, std::size_t output_channels,
           std::array<std::size_t, 2> kernel_size, std::mt19937& generator,
           std::array<std::size_t, 2> stride = {1, 1}, std::array<std::size_t, 2> padding = {0, 0},
           bool use_bias = true, Device device = Device::cpu());
    Conv2d(std::size_t input_channels, std::size_t output_channels,
           std::array<std::size_t, 2> kernel_size, std::mt19937& generator,
           std::array<std::size_t, 2> stride, PaddingMode padding, bool use_bias = true,
           Device device = Device::cpu());
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] Tensor& weight() noexcept;
    [[nodiscard]] Tensor& bias();
    [[nodiscard]] std::size_t input_channels() const noexcept;
    [[nodiscard]] std::size_t output_channels() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> kernel_size() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> stride() const noexcept;
    [[nodiscard]] std::array<std::size_t, 2> padding() const noexcept;
    [[nodiscard]] bool has_bias() const noexcept;
    [[nodiscard]] ModuleType type() const noexcept override;

  private:
    std::array<std::size_t, 2> stride_;
    std::array<std::size_t, 2> padding_;
    Tensor weight_;
    std::optional<Tensor> bias_;
};

class Sequential final : public Module {
  public:
    Sequential() = default;
    explicit Sequential(std::vector<std::unique_ptr<Module>> modules);
    Sequential(const Sequential&) = delete;
    Sequential& operator=(const Sequential&) = delete;
    Sequential(Sequential&&) noexcept = default;
    Sequential& operator=(Sequential&&) noexcept = default;
    Sequential& add(std::unique_ptr<Module> module);
    [[nodiscard]] Tensor forward(const Tensor& input) override;
    [[nodiscard]] std::vector<NamedParameter> named_parameters() override;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::vector<std::unique_ptr<Module>>& modules() const noexcept;
    [[nodiscard]] std::vector<NamedParameter> named_buffers() override;
    Module& to(const Device& device) override;
    Module& train(bool training = true) override;

  private:
    std::vector<std::unique_ptr<Module>> modules_;
};

void save_state_dict(Module& module, const std::filesystem::path& path);
void load_state_dict(Module& module, const std::filesystem::path& path);
void save_checkpoint(Sequential& model, const std::filesystem::path& path);
[[nodiscard]] std::unique_ptr<Sequential> load_checkpoint(const std::filesystem::path& path,
                                                          Device device = Device::cpu(),
                                                          std::uint32_t initialization_seed = 0);

} // namespace clnn::nn
