#include "clnn/nn/module.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace clnn::nn {
namespace detail {

struct ModuleHookState final {
    std::mutex mutex;
    std::uint64_t next_identifier = 1;
    std::vector<std::pair<std::uint64_t, ForwardHook>> hooks;
};

} // namespace detail

namespace {

constexpr char state_magic[] = {'C', 'L', 'N', 'N', 'A', 'U', 'T', '1'};
constexpr char checkpoint_magic[] = {'C', 'L', 'N', 'N', 'C', 'H', 'K', '2'};
constexpr std::uint32_t maximum_string_length = 1U << 20U;
constexpr std::uint32_t maximum_rank = 32;

std::array<std::size_t, 2> resolve_padding(const std::array<std::size_t, 2> kernel,
                                           const PaddingMode mode) {
    if (mode == PaddingMode::valid)
        return {0, 0};
    if (kernel[0] == 0 || kernel[1] == 0 || kernel[0] % 2 == 0 || kernel[1] % 2 == 0) {
        throw std::invalid_argument("same padding requires odd, non-zero convolution kernels");
    }
    return {kernel[0] / 2, kernel[1] / 2};
}

template <typename Value> void write_value(std::ostream& stream, const Value& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(Value));
    if (!stream) {
        throw std::runtime_error("failed while writing model state");
    }
}

template <typename Value> Value read_value(std::istream& stream) {
    Value value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!stream) {
        throw std::runtime_error("model state is truncated or corrupt");
    }
    return value;
}

} // namespace

ForwardHookHandle::ForwardHookHandle(std::weak_ptr<detail::ModuleHookState> state,
                                     const std::uint64_t identifier)
    : state_(std::move(state)), identifier_(identifier) {}

ForwardHookHandle::~ForwardHookHandle() {
    remove();
}

ForwardHookHandle::ForwardHookHandle(ForwardHookHandle&& other) noexcept
    : state_(std::move(other.state_)), identifier_(std::exchange(other.identifier_, 0)) {}

ForwardHookHandle& ForwardHookHandle::operator=(ForwardHookHandle&& other) noexcept {
    if (this != &other) {
        remove();
        state_ = std::move(other.state_);
        identifier_ = std::exchange(other.identifier_, 0);
    }
    return *this;
}

void ForwardHookHandle::remove() noexcept {
    if (identifier_ == 0)
        return;
    if (const auto state = state_.lock()) {
        std::scoped_lock lock(state->mutex);
        std::erase_if(state->hooks,
                      [this](const auto& hook) { return hook.first == identifier_; });
    }
    identifier_ = 0;
    state_.reset();
}

bool ForwardHookHandle::active() const noexcept {
    return identifier_ != 0 && !state_.expired();
}

Module::Module() : hook_state_(std::make_shared<detail::ModuleHookState>()) {}

Module::Module(const Module& other)
    : training_(other.training_), hook_state_(std::make_shared<detail::ModuleHookState>()) {}

Module& Module::operator=(const Module& other) {
    if (this != &other) {
        training_ = other.training_;
        hook_state_ = std::make_shared<detail::ModuleHookState>();
    }
    return *this;
}

std::vector<NamedParameter> Module::named_parameters() {
    return {};
}
std::vector<NamedParameter> Module::named_buffers() {
    return {};
}
ModuleType Module::type() const noexcept {
    return ModuleType::custom;
}

std::vector<Tensor*> Module::parameters() {
    const auto named = named_parameters();
    std::vector<Tensor*> result;
    result.reserve(named.size());
    for (const auto& parameter : named) {
        result.push_back(parameter.tensor);
    }
    return result;
}

Tensor Module::operator()(const Tensor& input) {
    auto output = forward(input);
    std::vector<ForwardHook> hooks;
    {
        std::scoped_lock lock(hook_state_->mutex);
        hooks.reserve(hook_state_->hooks.size());
        for (const auto& entry : hook_state_->hooks)
            hooks.push_back(entry.second);
    }
    for (const auto& hook : hooks)
        hook(*this, input, output);
    return output;
}

Module& Module::to(const Device& device) {
    for (auto* parameter : parameters()) {
        *parameter = parameter->to(device);
    }
    return *this;
}

Module& Module::train(const bool training) {
    training_ = training;
    return *this;
}

Module& Module::eval() {
    return train(false);
}

bool Module::training() const noexcept {
    return training_;
}

ForwardHookHandle Module::register_forward_hook(ForwardHook hook) {
    if (!hook)
        throw std::invalid_argument("forward hook cannot be empty");
    std::scoped_lock lock(hook_state_->mutex);
    const auto identifier = hook_state_->next_identifier++;
    hook_state_->hooks.emplace_back(identifier, std::move(hook));
    return ForwardHookHandle(hook_state_, identifier);
}

Linear::Linear(const std::size_t input_features, const std::size_t output_features,
               std::mt19937& generator, const bool use_bias, const Device device)
    : input_features_(input_features), output_features_(output_features),
      weight_(Tensor::randn(
          {input_features, output_features}, generator, 0.0F,
          std::sqrt(2.0F / static_cast<float>(std::max<std::size_t>(1, input_features))), true,
          "weight", device)) {
    if (input_features == 0 || output_features == 0) {
        throw std::invalid_argument("linear layer feature counts must be greater than zero");
    }
    if (use_bias) {
        bias_ = Tensor::zeros({output_features}, true, "bias", device);
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (input.ndim() != 2 || input.shape()[1] != input_features_) {
        throw std::invalid_argument("linear expected input shape [batch, " +
                                    std::to_string(input_features_) + "], got " +
                                    shape_string(input.shape()));
    }
    auto output = matmul(input, weight_);
    return bias_.has_value() ? output + *bias_ : output;
}

std::vector<NamedParameter> Linear::named_parameters() {
    std::vector<NamedParameter> result{{"weight", &weight_}};
    if (bias_.has_value()) {
        result.push_back({"bias", &*bias_});
    }
    return result;
}

Tensor& Linear::weight() noexcept {
    return weight_;
}

Tensor& Linear::bias() {
    if (!bias_.has_value()) {
        throw std::logic_error("this linear layer has no bias");
    }
    return *bias_;
}

std::size_t Linear::input_features() const noexcept {
    return input_features_;
}
std::size_t Linear::output_features() const noexcept {
    return output_features_;
}
bool Linear::has_bias() const noexcept {
    return bias_.has_value();
}
ModuleType Linear::type() const noexcept {
    return ModuleType::linear;
}

Tensor ReLU::forward(const Tensor& input) {
    return relu(input);
}
Tensor Sigmoid::forward(const Tensor& input) {
    return sigmoid(input);
}
Tensor Tanh::forward(const Tensor& input) {
    return tanh(input);
}
ModuleType ReLU::type() const noexcept {
    return ModuleType::relu;
}
ModuleType Sigmoid::type() const noexcept {
    return ModuleType::sigmoid;
}
ModuleType Tanh::type() const noexcept {
    return ModuleType::tanh;
}

LeakyReLU::LeakyReLU(const float negative_slope) : negative_slope_(negative_slope) {
    if (negative_slope < 0.0F)
        throw std::invalid_argument("leaky ReLU slope must be non-negative");
}
Tensor LeakyReLU::forward(const Tensor& input) {
    return leaky_relu(input, negative_slope_);
}
ModuleType LeakyReLU::type() const noexcept {
    return ModuleType::leaky_relu;
}
float LeakyReLU::negative_slope() const noexcept {
    return negative_slope_;
}

Softmax::Softmax(const std::size_t dimension) : dimension_(dimension) {}
Tensor Softmax::forward(const Tensor& input) {
    return softmax(input, dimension_);
}
ModuleType Softmax::type() const noexcept {
    return ModuleType::softmax;
}
std::size_t Softmax::dimension() const noexcept {
    return dimension_;
}

Tensor Flatten::forward(const Tensor& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument(
            "flatten expects a batch dimension and at least one feature dimension");
    }
    return reshape(input, {input.shape().front(), input.size() / input.shape().front()});
}
ModuleType Flatten::type() const noexcept {
    return ModuleType::flatten;
}

MaxPool2d::MaxPool2d(const std::array<std::size_t, 2> kernel_size,
                     const std::array<std::size_t, 2> stride,
                     const std::array<std::size_t, 2> padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size[0] == 0 || kernel_size[1] == 0)
        throw std::invalid_argument("pooling kernel dimensions must be non-zero");
}
Tensor MaxPool2d::forward(const Tensor& input) {
    return max_pool2d(input, kernel_size_, stride_, padding_);
}
ModuleType MaxPool2d::type() const noexcept {
    return ModuleType::max_pool_2d;
}
std::array<std::size_t, 2> MaxPool2d::kernel_size() const noexcept {
    return kernel_size_;
}
std::array<std::size_t, 2> MaxPool2d::stride() const noexcept {
    return stride_;
}
std::array<std::size_t, 2> MaxPool2d::padding() const noexcept {
    return padding_;
}

AvgPool2d::AvgPool2d(const std::array<std::size_t, 2> kernel_size,
                     const std::array<std::size_t, 2> stride,
                     const std::array<std::size_t, 2> padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size[0] == 0 || kernel_size[1] == 0)
        throw std::invalid_argument("pooling kernel dimensions must be non-zero");
}
Tensor AvgPool2d::forward(const Tensor& input) {
    return avg_pool2d(input, kernel_size_, stride_, padding_);
}
ModuleType AvgPool2d::type() const noexcept {
    return ModuleType::average_pool_2d;
}
std::array<std::size_t, 2> AvgPool2d::kernel_size() const noexcept {
    return kernel_size_;
}
std::array<std::size_t, 2> AvgPool2d::stride() const noexcept {
    return stride_;
}
std::array<std::size_t, 2> AvgPool2d::padding() const noexcept {
    return padding_;
}

Dropout::Dropout(const float probability, const std::uint32_t seed)
    : probability_(probability), generator_(seed) {
    if (probability < 0.0F || probability >= 1.0F)
        throw std::invalid_argument("dropout probability must be in [0, 1)");
}
Tensor Dropout::forward(const Tensor& input) {
    if (!training_ || probability_ == 0.0F)
        return input;
    const auto keep_probability = 1.0F - probability_;
    std::bernoulli_distribution keep(keep_probability);
    std::vector<float> mask(input.size());
    for (auto& value : mask)
        value = keep(generator_) ? 1.0F / keep_probability : 0.0F;
    return input * Tensor(std::move(mask), input.shape(), false, {}, input.device());
}
ModuleType Dropout::type() const noexcept {
    return ModuleType::dropout;
}
float Dropout::probability() const noexcept {
    return probability_;
}

BatchNorm::BatchNorm(const std::size_t features, const float epsilon, const float momentum,
                     const bool affine, const Device device)
    : features_(features), epsilon_(epsilon), momentum_(momentum),
      running_mean_(Tensor::zeros({features}, false, "running_mean", device)),
      running_variance_(Tensor::ones({features}, false, "running_variance", device)) {
    if (features == 0)
        throw std::invalid_argument("batch normalization requires at least one feature");
    if (epsilon <= 0.0F)
        throw std::invalid_argument("batch normalization epsilon must be positive");
    if (momentum < 0.0F || momentum > 1.0F)
        throw std::invalid_argument("batch normalization momentum must be in [0, 1]");
    if (affine) {
        weight_ = Tensor::ones({features}, true, "weight", device);
        bias_ = Tensor::zeros({features}, true, "bias", device);
    }
}

Tensor BatchNorm::forward(const Tensor& input) {
    if ((input.ndim() != 2 && input.ndim() != 4) || input.shape()[1] != features_)
        throw std::invalid_argument("batch normalization expects [N, C] or [N, C, H, W] input");
    Shape broadcast_shape(input.ndim(), 1);
    broadcast_shape[1] = features_;
    Tensor center;
    Tensor variance;
    if (training_) {
        center = mean(input, 0, true);
        if (input.ndim() == 4) {
            center = mean(center, 2, true);
            center = mean(center, 3, true);
        }
        const auto centered = input - center;
        variance = mean(centered * centered, 0, true);
        if (input.ndim() == 4) {
            variance = mean(variance, 2, true);
            variance = mean(variance, 3, true);
        }
        const auto batch_mean = reshape(center.detach(), {features_});
        const auto batch_variance = reshape(variance.detach(), {features_});
        running_mean_ = (running_mean_ * (1.0F - momentum_) + batch_mean * momentum_).detach();
        running_variance_ =
            (running_variance_ * (1.0F - momentum_) + batch_variance * momentum_).detach();
    } else {
        center = reshape(running_mean_, broadcast_shape);
        variance = reshape(running_variance_, broadcast_shape);
    }
    auto output = (input - center) / pow(variance + epsilon_, 0.5F);
    if (weight_.has_value()) {
        output = output * reshape(*weight_, broadcast_shape) + reshape(*bias_, broadcast_shape);
    }
    return output;
}

std::vector<NamedParameter> BatchNorm::named_parameters() {
    if (!weight_.has_value())
        return {};
    return {{"weight", &*weight_}, {"bias", &*bias_}};
}
std::vector<NamedParameter> BatchNorm::named_buffers() {
    return {{"running_mean", &running_mean_}, {"running_variance", &running_variance_}};
}
ModuleType BatchNorm::type() const noexcept {
    return ModuleType::batch_normalization;
}
Module& BatchNorm::to(const Device& device) {
    Module::to(device);
    running_mean_ = running_mean_.to(device);
    running_variance_ = running_variance_.to(device);
    return *this;
}
std::size_t BatchNorm::features() const noexcept {
    return features_;
}
float BatchNorm::epsilon() const noexcept {
    return epsilon_;
}
float BatchNorm::momentum() const noexcept {
    return momentum_;
}
bool BatchNorm::affine() const noexcept {
    return weight_.has_value();
}
Tensor& BatchNorm::weight() {
    if (!weight_.has_value())
        throw std::logic_error("this batch normalization is not affine");
    return *weight_;
}
Tensor& BatchNorm::bias() {
    if (!bias_.has_value())
        throw std::logic_error("this batch normalization is not affine");
    return *bias_;
}
const Tensor& BatchNorm::running_mean() const noexcept {
    return running_mean_;
}
const Tensor& BatchNorm::running_variance() const noexcept {
    return running_variance_;
}

LayerNorm::LayerNorm(Shape normalized_shape, const float epsilon, const bool affine,
                     const Device device)
    : normalized_shape_(std::move(normalized_shape)), epsilon_(epsilon) {
    if (normalized_shape_.empty() ||
        std::any_of(normalized_shape_.begin(), normalized_shape_.end(),
                    [](const std::size_t dimension) { return dimension == 0; }))
        throw std::invalid_argument("layer normalization shape must be non-empty and non-zero");
    if (epsilon <= 0.0F)
        throw std::invalid_argument("layer normalization epsilon must be positive");
    if (affine) {
        weight_ = Tensor::ones(normalized_shape_, true, "weight", device);
        bias_ = Tensor::zeros(normalized_shape_, true, "bias", device);
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    if (input.ndim() < normalized_shape_.size() ||
        !std::equal(normalized_shape_.rbegin(), normalized_shape_.rend(), input.shape().rbegin()))
        throw std::invalid_argument("layer normalization input trailing dimensions do not match");
    const auto first_dimension = input.ndim() - normalized_shape_.size();
    auto center = input;
    for (std::size_t dimension = first_dimension; dimension < input.ndim(); ++dimension)
        center = mean(center, dimension, true);
    const auto centered = input - center;
    auto variance = centered * centered;
    for (std::size_t dimension = first_dimension; dimension < input.ndim(); ++dimension)
        variance = mean(variance, dimension, true);
    auto output = centered / pow(variance + epsilon_, 0.5F);
    if (weight_.has_value()) {
        Shape broadcast_shape(first_dimension, 1);
        broadcast_shape.insert(broadcast_shape.end(), normalized_shape_.begin(),
                               normalized_shape_.end());
        output = output * reshape(*weight_, broadcast_shape) + reshape(*bias_, broadcast_shape);
    }
    return output;
}
std::vector<NamedParameter> LayerNorm::named_parameters() {
    if (!weight_.has_value())
        return {};
    return {{"weight", &*weight_}, {"bias", &*bias_}};
}
ModuleType LayerNorm::type() const noexcept {
    return ModuleType::layer_normalization;
}
const Shape& LayerNorm::normalized_shape() const noexcept {
    return normalized_shape_;
}
float LayerNorm::epsilon() const noexcept {
    return epsilon_;
}
bool LayerNorm::affine() const noexcept {
    return weight_.has_value();
}
Tensor& LayerNorm::weight() {
    if (!weight_.has_value())
        throw std::logic_error("this layer normalization is not affine");
    return *weight_;
}
Tensor& LayerNorm::bias() {
    if (!bias_.has_value())
        throw std::logic_error("this layer normalization is not affine");
    return *bias_;
}

Tensor GlobalAvgPool2d::forward(const Tensor& input) {
    if (input.ndim() != 4)
        throw std::invalid_argument("global average pooling expects input shape [N, C, H, W]");
    return mean(mean(input, 2), 2);
}
ModuleType GlobalAvgPool2d::type() const noexcept {
    return ModuleType::global_average_pool_2d;
}

Residual::Residual(std::unique_ptr<Module> module) : module_(std::move(module)) {
    if (module_ == nullptr)
        throw std::invalid_argument("residual module cannot be null");
}
Tensor Residual::forward(const Tensor& input) {
    return input + (*module_)(input);
}
std::vector<NamedParameter> Residual::named_parameters() {
    auto result = module_->named_parameters();
    for (auto& parameter : result)
        parameter.name = "module." + parameter.name;
    return result;
}
std::vector<NamedParameter> Residual::named_buffers() {
    auto result = module_->named_buffers();
    for (auto& buffer : result)
        buffer.name = "module." + buffer.name;
    return result;
}
ModuleType Residual::type() const noexcept {
    return ModuleType::residual;
}
Module& Residual::to(const Device& device) {
    module_->to(device);
    return *this;
}
Module& Residual::train(const bool training) {
    training_ = training;
    module_->train(training);
    return *this;
}
Module& Residual::module() noexcept {
    return *module_;
}

Conv2d::Conv2d(const std::size_t input_channels, const std::size_t output_channels,
               const std::array<std::size_t, 2> kernel_size, std::mt19937& generator,
               const std::array<std::size_t, 2> stride, const std::array<std::size_t, 2> padding,
               const bool use_bias, const Device device)
    : stride_(stride), padding_(padding),
      weight_(Tensor::randn(
          {output_channels, input_channels, kernel_size[0], kernel_size[1]}, generator, 0.0F,
          std::sqrt(2.0F / static_cast<float>(std::max<std::size_t>(
                               1, input_channels * kernel_size[0] * kernel_size[1]))),
          true, "weight", device)) {
    if (input_channels == 0 || output_channels == 0 || kernel_size[0] == 0 || kernel_size[1] == 0 ||
        stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument("convolution channels, kernel, and stride must be non-zero");
    }
    if (use_bias) {
        bias_ = Tensor::zeros({output_channels}, true, "bias", device);
    }
}

Conv2d::Conv2d(const std::size_t input_channels, const std::size_t output_channels,
               const std::array<std::size_t, 2> kernel_size, std::mt19937& generator,
               const std::array<std::size_t, 2> stride, const PaddingMode padding,
               const bool use_bias, const Device device)
    : Conv2d(input_channels, output_channels, kernel_size, generator, stride,
             resolve_padding(kernel_size, padding), use_bias, device) {}

Tensor Conv2d::forward(const Tensor& input) {
    return conv2d(input, weight_, bias_, stride_, padding_);
}

std::vector<NamedParameter> Conv2d::named_parameters() {
    std::vector<NamedParameter> result{{"weight", &weight_}};
    if (bias_.has_value()) {
        result.push_back({"bias", &*bias_});
    }
    return result;
}

Tensor& Conv2d::weight() noexcept {
    return weight_;
}

Tensor& Conv2d::bias() {
    if (!bias_.has_value()) {
        throw std::logic_error("this convolution has no bias");
    }
    return *bias_;
}

std::size_t Conv2d::input_channels() const noexcept {
    return weight_.shape()[1];
}
std::size_t Conv2d::output_channels() const noexcept {
    return weight_.shape()[0];
}
std::array<std::size_t, 2> Conv2d::kernel_size() const noexcept {
    return {weight_.shape()[2], weight_.shape()[3]};
}
std::array<std::size_t, 2> Conv2d::stride() const noexcept {
    return stride_;
}
std::array<std::size_t, 2> Conv2d::padding() const noexcept {
    return padding_;
}
bool Conv2d::has_bias() const noexcept {
    return bias_.has_value();
}
ModuleType Conv2d::type() const noexcept {
    return ModuleType::convolution_2d;
}

Sequential::Sequential(std::vector<std::unique_ptr<Module>> modules)
    : modules_(std::move(modules)) {
    for (const auto& module : modules_) {
        if (module == nullptr) {
            throw std::invalid_argument("sequential cannot contain a null module");
        }
    }
}

Sequential& Sequential::add(std::unique_ptr<Module> module) {
    if (module == nullptr) {
        throw std::invalid_argument("cannot add a null module to sequential");
    }
    modules_.push_back(std::move(module));
    return *this;
}

Tensor Sequential::forward(const Tensor& input) {
    auto output = input;
    for (const auto& module : modules_) {
        output = (*module)(output);
    }
    return output;
}

std::vector<NamedParameter> Sequential::named_parameters() {
    std::vector<NamedParameter> result;
    for (std::size_t index = 0; index < modules_.size(); ++index) {
        for (auto parameter : modules_[index]->named_parameters()) {
            parameter.name = std::to_string(index) + "." + parameter.name;
            result.push_back(std::move(parameter));
        }
    }
    return result;
}

std::vector<NamedParameter> Sequential::named_buffers() {
    std::vector<NamedParameter> result;
    for (std::size_t index = 0; index < modules_.size(); ++index) {
        for (auto buffer : modules_[index]->named_buffers()) {
            buffer.name = std::to_string(index) + "." + buffer.name;
            result.push_back(std::move(buffer));
        }
    }
    return result;
}

std::size_t Sequential::size() const noexcept {
    return modules_.size();
}
const std::vector<std::unique_ptr<Module>>& Sequential::modules() const noexcept {
    return modules_;
}

Module& Sequential::to(const Device& device) {
    for (auto& module : modules_)
        module->to(device);
    return *this;
}

Module& Sequential::train(const bool training) {
    training_ = training;
    for (auto& module : modules_)
        module->train(training);
    return *this;
}

namespace {

std::vector<NamedParameter> state_tensors(Module& module) {
    auto result = module.named_parameters();
    auto buffers = module.named_buffers();
    result.insert(result.end(), std::make_move_iterator(buffers.begin()),
                  std::make_move_iterator(buffers.end()));
    return result;
}

} // namespace

void save_state_dict(Module& module, const std::filesystem::path& path) {
    auto parameters = state_tensors(module);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open model state for writing: " + path.string());
    }
    stream.write(state_magic, sizeof(state_magic));
    write_value(stream, static_cast<std::uint64_t>(parameters.size()));
    for (const auto& parameter : parameters) {
        if (parameter.tensor == nullptr || !parameter.tensor->defined()) {
            throw std::logic_error("module returned an invalid parameter");
        }
        if (parameter.name.size() > maximum_string_length) {
            throw std::length_error("parameter name is too long to serialize");
        }
        write_value(stream, static_cast<std::uint32_t>(parameter.name.size()));
        stream.write(parameter.name.data(), static_cast<std::streamsize>(parameter.name.size()));
        write_value(stream, static_cast<std::uint32_t>(parameter.tensor->ndim()));
        for (const auto dimension : parameter.tensor->shape()) {
            write_value(stream, static_cast<std::uint64_t>(dimension));
        }
        write_value(stream, static_cast<std::uint64_t>(parameter.tensor->size()));
        stream.write(reinterpret_cast<const char*>(parameter.tensor->data().data()),
                     static_cast<std::streamsize>(parameter.tensor->size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("failed while writing parameter '" + parameter.name + "'");
        }
    }
}

void load_state_dict(Module& module, const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open model state for reading: " + path.string());
    }
    char magic[sizeof(state_magic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || !std::equal(std::begin(magic), std::end(magic), std::begin(state_magic))) {
        throw std::runtime_error("file is not a supported CLNN model state");
    }
    const auto count = read_value<std::uint64_t>(stream);
    if (count > 1'000'000ULL) {
        throw std::runtime_error("model state contains an unreasonable parameter count");
    }

    struct Record final {
        Shape shape;
        std::vector<float> data;
    };
    std::unordered_map<std::string, Record> records;
    for (std::uint64_t entry = 0; entry < count; ++entry) {
        const auto name_length = read_value<std::uint32_t>(stream);
        if (name_length > maximum_string_length) {
            throw std::runtime_error("model state contains an unreasonable name length");
        }
        std::string name(name_length, '\0');
        stream.read(name.data(), static_cast<std::streamsize>(name.size()));
        const auto rank = read_value<std::uint32_t>(stream);
        if (rank > maximum_rank) {
            throw std::runtime_error("model state contains an unreasonable tensor rank");
        }
        Shape shape(rank);
        for (auto& dimension : shape) {
            const auto saved_dimension = read_value<std::uint64_t>(stream);
            if (saved_dimension > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("model tensor dimension is too large");
            }
            dimension = static_cast<std::size_t>(saved_dimension);
        }
        const auto element_count = read_value<std::uint64_t>(stream);
        if (element_count != numel(shape)) {
            throw std::runtime_error("model tensor shape and element count disagree");
        }
        std::vector<float> data(static_cast<std::size_t>(element_count));
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("model state is truncated while reading '" + name + "'");
        }
        if (!records.emplace(std::move(name), Record{std::move(shape), std::move(data)}).second) {
            throw std::runtime_error("model state contains duplicate parameter names");
        }
    }

    const auto parameters = state_tensors(module);
    if (records.size() != parameters.size()) {
        throw std::runtime_error("model state parameter count does not match the module");
    }
    for (const auto& parameter : parameters) {
        const auto record = records.find(parameter.name);
        if (record == records.end()) {
            throw std::runtime_error("model state is missing parameter '" + parameter.name + "'");
        }
        if (record->second.shape != parameter.tensor->shape()) {
            throw std::runtime_error("shape mismatch for parameter '" + parameter.name + "'");
        }
    }
    for (const auto& parameter : parameters) {
        parameter.tensor->set_data(records.at(parameter.name).data);
    }
}

void save_checkpoint(Sequential& model, const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot open checkpoint for writing: " + path.string());
    stream.write(checkpoint_magic, sizeof(checkpoint_magic));
    write_value(stream, static_cast<std::uint64_t>(model.size()));
    for (const auto& module : model.modules()) {
        write_value(stream, static_cast<std::uint32_t>(module->type()));
        switch (module->type()) {
        case ModuleType::linear: {
            const auto& linear = dynamic_cast<const Linear&>(*module);
            write_value(stream, static_cast<std::uint64_t>(linear.input_features()));
            write_value(stream, static_cast<std::uint64_t>(linear.output_features()));
            write_value(stream, static_cast<std::uint8_t>(linear.has_bias()));
            break;
        }
        case ModuleType::convolution_2d: {
            const auto& convolution = dynamic_cast<const Conv2d&>(*module);
            write_value(stream, static_cast<std::uint64_t>(convolution.input_channels()));
            write_value(stream, static_cast<std::uint64_t>(convolution.output_channels()));
            for (const auto value : convolution.kernel_size())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : convolution.stride())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : convolution.padding())
                write_value(stream, static_cast<std::uint64_t>(value));
            write_value(stream, static_cast<std::uint8_t>(convolution.has_bias()));
            break;
        }
        case ModuleType::relu:
        case ModuleType::sigmoid:
        case ModuleType::tanh:
        case ModuleType::flatten:
            break;
        case ModuleType::leaky_relu: {
            const auto& activation = dynamic_cast<const LeakyReLU&>(*module);
            write_value(stream, activation.negative_slope());
            break;
        }
        case ModuleType::softmax: {
            const auto& activation = dynamic_cast<const Softmax&>(*module);
            write_value(stream, static_cast<std::uint64_t>(activation.dimension()));
            break;
        }
        case ModuleType::max_pool_2d: {
            const auto& pooling = dynamic_cast<const MaxPool2d&>(*module);
            for (const auto value : pooling.kernel_size())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : pooling.stride())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : pooling.padding())
                write_value(stream, static_cast<std::uint64_t>(value));
            break;
        }
        case ModuleType::average_pool_2d: {
            const auto& pooling = dynamic_cast<const AvgPool2d&>(*module);
            for (const auto value : pooling.kernel_size())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : pooling.stride())
                write_value(stream, static_cast<std::uint64_t>(value));
            for (const auto value : pooling.padding())
                write_value(stream, static_cast<std::uint64_t>(value));
            break;
        }
        case ModuleType::dropout: {
            const auto& dropout = dynamic_cast<const Dropout&>(*module);
            write_value(stream, dropout.probability());
            break;
        }
        case ModuleType::batch_normalization: {
            const auto& normalization = dynamic_cast<const BatchNorm&>(*module);
            write_value(stream, static_cast<std::uint64_t>(normalization.features()));
            write_value(stream, normalization.epsilon());
            write_value(stream, normalization.momentum());
            write_value(stream, static_cast<std::uint8_t>(normalization.affine()));
            break;
        }
        case ModuleType::layer_normalization: {
            const auto& normalization = dynamic_cast<const LayerNorm&>(*module);
            write_value(stream,
                        static_cast<std::uint32_t>(normalization.normalized_shape().size()));
            for (const auto dimension : normalization.normalized_shape())
                write_value(stream, static_cast<std::uint64_t>(dimension));
            write_value(stream, normalization.epsilon());
            write_value(stream, static_cast<std::uint8_t>(normalization.affine()));
            break;
        }
        case ModuleType::global_average_pool_2d:
            break;
        default:
            throw std::runtime_error("checkpoint cannot serialize a custom or nested module");
        }
    }

    const auto parameters = state_tensors(model);
    write_value(stream, static_cast<std::uint64_t>(parameters.size()));
    for (const auto& parameter : parameters) {
        write_value(stream, static_cast<std::uint32_t>(parameter.name.size()));
        stream.write(parameter.name.data(), static_cast<std::streamsize>(parameter.name.size()));
        write_value(stream, static_cast<std::uint32_t>(parameter.tensor->ndim()));
        for (const auto dimension : parameter.tensor->shape()) {
            write_value(stream, static_cast<std::uint64_t>(dimension));
        }
        write_value(stream, static_cast<std::uint64_t>(parameter.tensor->size()));
        stream.write(reinterpret_cast<const char*>(parameter.tensor->data().data()),
                     static_cast<std::streamsize>(parameter.tensor->size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("failed while writing checkpoint parameters");
    }
}

std::unique_ptr<Sequential> load_checkpoint(const std::filesystem::path& path, const Device device,
                                            const std::uint32_t initialization_seed) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open checkpoint for reading: " + path.string());
    char magic[sizeof(checkpoint_magic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || !std::equal(std::begin(magic), std::end(magic), std::begin(checkpoint_magic))) {
        throw std::runtime_error("file is not a supported CLNN checkpoint");
    }
    const auto module_count = read_value<std::uint64_t>(stream);
    if (module_count > 100'000ULL)
        throw std::runtime_error("checkpoint module count is unreasonable");
    auto model = std::make_unique<Sequential>();
    std::mt19937 generator(initialization_seed);
    for (std::uint64_t index = 0; index < module_count; ++index) {
        const auto type = static_cast<ModuleType>(read_value<std::uint32_t>(stream));
        switch (type) {
        case ModuleType::linear: {
            const auto input = read_value<std::uint64_t>(stream);
            const auto output = read_value<std::uint64_t>(stream);
            const auto bias = read_value<std::uint8_t>(stream) != 0;
            model->add(std::make_unique<Linear>(static_cast<std::size_t>(input),
                                                static_cast<std::size_t>(output), generator, bias,
                                                device));
            break;
        }
        case ModuleType::convolution_2d: {
            const auto input = read_value<std::uint64_t>(stream);
            const auto output = read_value<std::uint64_t>(stream);
            std::array<std::size_t, 2> kernel{}, stride{}, padding{};
            for (auto& value : kernel)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            for (auto& value : stride)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            for (auto& value : padding)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            const auto bias = read_value<std::uint8_t>(stream) != 0;
            model->add(std::make_unique<Conv2d>(static_cast<std::size_t>(input),
                                                static_cast<std::size_t>(output), kernel, generator,
                                                stride, padding, bias, device));
            break;
        }
        case ModuleType::relu:
            model->add(std::make_unique<ReLU>());
            break;
        case ModuleType::sigmoid:
            model->add(std::make_unique<Sigmoid>());
            break;
        case ModuleType::tanh:
            model->add(std::make_unique<Tanh>());
            break;
        case ModuleType::flatten:
            model->add(std::make_unique<Flatten>());
            break;
        case ModuleType::leaky_relu:
            model->add(std::make_unique<LeakyReLU>(read_value<float>(stream)));
            break;
        case ModuleType::softmax:
            model->add(std::make_unique<Softmax>(
                static_cast<std::size_t>(read_value<std::uint64_t>(stream))));
            break;
        case ModuleType::max_pool_2d:
        case ModuleType::average_pool_2d: {
            std::array<std::size_t, 2> kernel{}, stride{}, padding{};
            for (auto& value : kernel)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            for (auto& value : stride)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            for (auto& value : padding)
                value = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            if (type == ModuleType::max_pool_2d)
                model->add(std::make_unique<MaxPool2d>(kernel, stride, padding));
            else
                model->add(std::make_unique<AvgPool2d>(kernel, stride, padding));
            break;
        }
        case ModuleType::dropout:
            model->add(
                std::make_unique<Dropout>(read_value<float>(stream),
                                          initialization_seed + static_cast<std::uint32_t>(index)));
            break;
        case ModuleType::batch_normalization: {
            const auto features = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            const auto epsilon = read_value<float>(stream);
            const auto momentum = read_value<float>(stream);
            const auto affine = read_value<std::uint8_t>(stream) != 0;
            model->add(std::make_unique<BatchNorm>(features, epsilon, momentum, affine, device));
            break;
        }
        case ModuleType::layer_normalization: {
            const auto rank = read_value<std::uint32_t>(stream);
            if (rank == 0 || rank > maximum_rank)
                throw std::runtime_error("checkpoint layer normalization rank is invalid");
            Shape normalized_shape(rank);
            for (auto& dimension : normalized_shape)
                dimension = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            const auto epsilon = read_value<float>(stream);
            const auto affine = read_value<std::uint8_t>(stream) != 0;
            model->add(
                std::make_unique<LayerNorm>(std::move(normalized_shape), epsilon, affine, device));
            break;
        }
        case ModuleType::global_average_pool_2d:
            model->add(std::make_unique<GlobalAvgPool2d>());
            break;
        default:
            throw std::runtime_error("checkpoint contains an unsupported module type");
        }
    }

    struct Record final {
        Shape shape;
        std::vector<float> data;
    };
    std::unordered_map<std::string, Record> records;
    const auto parameter_count = read_value<std::uint64_t>(stream);
    if (parameter_count > 1'000'000ULL)
        throw std::runtime_error("checkpoint parameter count is unreasonable");
    for (std::uint64_t entry = 0; entry < parameter_count; ++entry) {
        const auto name_length = read_value<std::uint32_t>(stream);
        if (name_length > maximum_string_length)
            throw std::runtime_error("checkpoint parameter name is too long");
        std::string name(name_length, '\0');
        stream.read(name.data(), static_cast<std::streamsize>(name.size()));
        const auto rank = read_value<std::uint32_t>(stream);
        if (rank > maximum_rank)
            throw std::runtime_error("checkpoint tensor rank is unreasonable");
        Shape shape(rank);
        for (auto& dimension : shape)
            dimension = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
        const auto elements = read_value<std::uint64_t>(stream);
        if (elements != numel(shape))
            throw std::runtime_error("checkpoint tensor shape is corrupt");
        std::vector<float> values(static_cast<std::size_t>(elements));
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!stream ||
            !records.emplace(std::move(name), Record{std::move(shape), std::move(values)}).second) {
            throw std::runtime_error("checkpoint parameters are corrupt or duplicated");
        }
    }
    const auto parameters = state_tensors(*model);
    if (parameters.size() != records.size())
        throw std::runtime_error("checkpoint parameter count does not match architecture");
    for (const auto& parameter : parameters) {
        const auto found = records.find(parameter.name);
        if (found == records.end() || found->second.shape != parameter.tensor->shape()) {
            throw std::runtime_error("checkpoint parameter mismatch for '" + parameter.name + "'");
        }
    }
    for (const auto& parameter : parameters)
        parameter.tensor->set_data(records.at(parameter.name).data);
    return model;
}

} // namespace clnn::nn
