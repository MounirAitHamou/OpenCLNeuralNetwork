#include "clnn/autograd/tensor.hpp"

#include "opencl/backend.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace clnn {

namespace {
thread_local bool gradient_recording_enabled = true;
thread_local detail::TensorImpl* gradient_leaf_target = nullptr;
}

namespace detail {

struct GraphNode final {
    std::string operation;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::vector<std::uint64_t> parent_versions;
    std::function<void(const Tensor&)> backward;
};

struct TensorImpl final {
    mutable std::vector<float> data;
    Shape shape;
    Device device = Device::cpu();
    std::shared_ptr<opencl::Buffer> opencl_buffer;
    bool requires_grad = false;
    std::string name;
    std::vector<float> gradient;
    std::shared_ptr<opencl::Buffer> gradient_buffer;
    std::shared_ptr<GraphNode> grad_fn;
    std::uint64_t version = 0;
    bool graph_freed = false;
};

} // namespace detail

std::size_t numel(const Shape& shape) {
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument("tensor dimensions must be greater than zero");
        }
        if (result > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor size overflow");
        }
        result *= dimension;
    }
    return result;
}

std::string shape_string(const Shape& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

Tensor::Tensor(std::vector<float> data, Shape shape, const bool requires_grad, std::string name,
               const Device device)
    : implementation_(std::make_shared<detail::TensorImpl>()) {
    const auto expected_size = numel(shape);
    if (data.size() != expected_size) {
        throw std::invalid_argument("tensor data has " + std::to_string(data.size()) +
                                    " elements, but shape " + shape_string(shape) + " requires " +
                                    std::to_string(expected_size));
    }
    implementation_->data = std::move(data);
    implementation_->shape = std::move(shape);
    implementation_->device = device;
    if (device.type() == DeviceType::opencl) {
        implementation_->opencl_buffer = std::make_shared<opencl::Buffer>(
            device, implementation_->data.size(), implementation_->data.data());
    }
    implementation_->requires_grad = requires_grad;
    implementation_->name = std::move(name);
}

Tensor::Tensor(std::shared_ptr<detail::TensorImpl> implementation)
    : implementation_(std::move(implementation)) {}

Tensor Tensor::scalar(const float value, const bool requires_grad, std::string name,
                      const Device device) {
    return Tensor({value}, {}, requires_grad, std::move(name), device);
}

Tensor Tensor::zeros(const Shape& shape, const bool requires_grad, std::string name,
                     const Device device) {
    return Tensor(std::vector<float>(numel(shape), 0.0F), shape, requires_grad, std::move(name),
                  device);
}

Tensor Tensor::ones(const Shape& shape, const bool requires_grad, std::string name,
                    const Device device) {
    return Tensor(std::vector<float>(numel(shape), 1.0F), shape, requires_grad, std::move(name),
                  device);
}

Tensor Tensor::randn(const Shape& shape, std::mt19937& generator, const float mean,
                     const float standard_deviation, const bool requires_grad, std::string name,
                     const Device device) {
    if (!(standard_deviation > 0.0F)) {
        throw std::invalid_argument("standard deviation must be positive");
    }
    std::normal_distribution<float> distribution(mean, standard_deviation);
    std::vector<float> values(numel(shape));
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
    return Tensor(std::move(values), shape, requires_grad, std::move(name), device);
}

const std::shared_ptr<detail::TensorImpl>& Tensor::impl() const {
    if (!implementation_) {
        throw std::logic_error("operation attempted on an undefined tensor");
    }
    return implementation_;
}

bool Tensor::defined() const noexcept {
    return static_cast<bool>(implementation_);
}
const Shape& Tensor::shape() const {
    return impl()->shape;
}
std::size_t Tensor::ndim() const {
    return shape().size();
}
std::size_t Tensor::size() const {
    return numel(shape());
}
bool Tensor::is_scalar() const {
    return shape().empty();
}

float Tensor::item() const {
    if (!is_scalar()) {
        throw std::logic_error("item() requires a scalar tensor, got " + shape_string(shape()));
    }
    return data().front();
}

const std::vector<float>& Tensor::data() const {
    auto implementation = impl();
    if (implementation->opencl_buffer != nullptr) {
        implementation->data = implementation->opencl_buffer->read();
    }
    return implementation->data;
}
const std::vector<float>& Tensor::grad() const {
    auto implementation = impl();
    if (implementation->gradient_buffer != nullptr) {
        implementation->gradient = implementation->gradient_buffer->read();
    }
    return implementation->gradient;
}
bool Tensor::has_grad() const {
    const auto implementation = impl();
    return !implementation->gradient.empty() || implementation->gradient_buffer != nullptr;
}
bool Tensor::requires_grad() const {
    return impl()->requires_grad;
}
bool Tensor::is_leaf() const {
    return impl()->grad_fn == nullptr && !impl()->graph_freed;
}
const std::string& Tensor::name() const {
    return impl()->name;
}
Device Tensor::device() const {
    return impl()->device;
}

void Tensor::zero_grad() {
    auto implementation = impl();
    implementation->gradient.clear();
    implementation->gradient_buffer.reset();
}

void Tensor::set_data(std::vector<float> data) {
    auto implementation = impl();
    if (data.size() != implementation->data.size()) {
        throw std::invalid_argument("replacement data must preserve tensor shape");
    }
    if (implementation->grad_fn != nullptr || implementation->graph_freed) {
        throw std::logic_error("only leaf tensors may be updated in place");
    }
    implementation->data = std::move(data);
    if (implementation->opencl_buffer != nullptr) {
        implementation->opencl_buffer->write(implementation->data);
    }
    ++implementation->version;
}

Tensor Tensor::detach() const {
    return Tensor(data(), shape(), false, name(), device());
}

Tensor Tensor::to(const Device& target) const {
    return Tensor(data(), shape(), requires_grad(), name(), target);
}

void detail_accumulate(const Tensor& tensor, const std::vector<float>& gradient) {
    auto implementation = tensor.impl();
    if (!implementation->requires_grad) {
        return;
    }
    if (gradient_leaf_target != nullptr && implementation->grad_fn == nullptr &&
        implementation.get() != gradient_leaf_target) {
        return;
    }
    if (gradient.size() != numel(implementation->shape)) {
        throw std::logic_error("internal autograd gradient size mismatch");
    }
    if (implementation->device.type() == DeviceType::opencl) {
        detail_accumulate(
            tensor, Tensor(gradient, implementation->shape, false, {}, implementation->device));
        return;
    }
    if (implementation->gradient.empty()) {
        implementation->gradient = gradient;
        return;
    }
    for (std::size_t index = 0; index < gradient.size(); ++index) {
        implementation->gradient[index] += gradient[index];
    }
}

void detail_accumulate(const Tensor& tensor, const Tensor& gradient) {
    auto implementation = tensor.impl();
    if (!implementation->requires_grad) {
        return;
    }
    if (gradient_leaf_target != nullptr && implementation->grad_fn == nullptr &&
        implementation.get() != gradient_leaf_target) {
        return;
    }
    if (gradient.shape() != implementation->shape) {
        throw std::logic_error("internal autograd gradient shape mismatch");
    }
    if (gradient.device() != implementation->device) {
        throw std::logic_error("internal autograd gradient device mismatch");
    }
    if (implementation->device.type() == DeviceType::cpu) {
        detail_accumulate(tensor, gradient.data());
        return;
    }
    implementation->gradient.clear();
    if (implementation->gradient_buffer == nullptr) {
        implementation->gradient_buffer = opencl::clone(*detail_opencl_buffer(gradient));
    } else {
        opencl::accumulate(*implementation->gradient_buffer, *detail_opencl_buffer(gradient));
    }
}

Tensor detail_make_result(std::vector<float> data, Shape shape, std::vector<Tensor> parents,
                          std::function<void(const Tensor&)> backward, std::string operation) {
    bool requires_grad = false;
    Device device = parents.empty() ? Device::cpu() : parents.front().device();
    for (const auto& parent : parents) {
        if (parent.device() != device) {
            throw std::invalid_argument("all tensors in an operation must be on the same device");
        }
    }
    if (grad_enabled()) {
        requires_grad = std::any_of(parents.begin(), parents.end(),
                                    [](const Tensor& parent) { return parent.requires_grad(); });
    }
    Tensor result(std::move(data), std::move(shape), requires_grad, {}, device);
    if (!requires_grad) {
        return result;
    }

    auto node = std::make_shared<detail::GraphNode>();
    node->operation = std::move(operation);
    node->backward = std::move(backward);
    node->parents.reserve(parents.size());
    node->parent_versions.reserve(parents.size());
    for (const auto& parent : parents) {
        node->parents.push_back(parent.impl());
        node->parent_versions.push_back(parent.impl()->version);
    }
    result.impl()->grad_fn = std::move(node);
    return result;
}

std::shared_ptr<opencl::Buffer> detail_opencl_buffer(const Tensor& tensor) {
    const auto implementation = tensor.impl();
    if (implementation->opencl_buffer == nullptr) {
        throw std::logic_error("tensor is not stored on an OpenCL device");
    }
    return implementation->opencl_buffer;
}

std::shared_ptr<opencl::Buffer> detail_opencl_gradient_buffer(const Tensor& tensor) {
    const auto implementation = tensor.impl();
    if (implementation->device.type() != DeviceType::opencl ||
        implementation->gradient_buffer == nullptr) {
        throw std::logic_error("tensor does not have an OpenCL gradient buffer");
    }
    return implementation->gradient_buffer;
}

Tensor detail_make_opencl_result(std::shared_ptr<opencl::Buffer> buffer, Shape shape,
                                 std::vector<Tensor> parents,
                                 std::function<void(const Tensor&)> backward, std::string operation,
                                 const Device device) {
    if (buffer == nullptr || buffer->size() != numel(shape) || buffer->device() != device) {
        throw std::invalid_argument("invalid OpenCL operation result buffer");
    }
    bool requires_grad =
        grad_enabled() && std::any_of(parents.begin(), parents.end(),
                                      [](const Tensor& parent) { return parent.requires_grad(); });
    auto implementation = std::make_shared<detail::TensorImpl>();
    implementation->shape = std::move(shape);
    implementation->device = device;
    implementation->requires_grad = requires_grad;
    implementation->opencl_buffer = std::move(buffer);
    Tensor result(std::move(implementation));
    if (!requires_grad)
        return result;
    auto node = std::make_shared<detail::GraphNode>();
    node->operation = std::move(operation);
    node->backward = std::move(backward);
    for (const auto& parent : parents) {
        if (parent.device() != device)
            throw std::invalid_argument("OpenCL operation device mismatch");
        node->parents.push_back(parent.impl());
        node->parent_versions.push_back(parent.impl()->version);
    }
    result.impl()->grad_fn = std::move(node);
    return result;
}

void detail_mark_opencl_modified(Tensor& tensor) {
    auto implementation = tensor.impl();
    if (implementation->opencl_buffer == nullptr || implementation->grad_fn != nullptr) {
        throw std::logic_error("only OpenCL leaf tensors may be modified by an optimizer");
    }
    ++implementation->version;
}

Tensor detail_clone_leaf(const Tensor& tensor, const bool requires_grad, std::string name) {
    const auto source = tensor.impl();
    auto implementation = std::make_shared<detail::TensorImpl>();
    implementation->shape = source->shape;
    implementation->device = source->device;
    implementation->requires_grad = requires_grad;
    implementation->name = name.empty() ? source->name : std::move(name);
    if (source->opencl_buffer != nullptr) {
        implementation->opencl_buffer = opencl::clone(*source->opencl_buffer);
    } else {
        implementation->data = source->data;
    }
    return Tensor(std::move(implementation));
}

void detail_backward_to(Tensor output, const Tensor& input, std::optional<Tensor> gradient) {
    const auto input_implementation = input.impl();
    if (!input_implementation->requires_grad || input_implementation->grad_fn != nullptr ||
        input_implementation->graph_freed) {
        throw std::invalid_argument(
            "selected backward input must be a live gradient-requiring leaf");
    }
    if (gradient_leaf_target != nullptr) {
        throw std::logic_error("selected backward passes cannot be nested");
    }
    struct CaptureGuard final {
        ~CaptureGuard() { gradient_leaf_target = nullptr; }
    } guard;
    gradient_leaf_target = input_implementation.get();
    output.backward(std::move(gradient));
}

Tensor detail_gradient_tensor(const Tensor& tensor) {
    const auto source = tensor.impl();
    if (source->gradient.empty() && source->gradient_buffer == nullptr) {
        throw std::logic_error("the requested tensor has no gradient");
    }
    auto implementation = std::make_shared<detail::TensorImpl>();
    implementation->shape = source->shape;
    implementation->device = source->device;
    if (source->gradient_buffer != nullptr) {
        implementation->opencl_buffer = source->gradient_buffer;
    } else {
        implementation->data = source->gradient;
    }
    return Tensor(std::move(implementation));
}

void Tensor::backward(std::optional<Tensor> gradient, const bool retain_graph) {
    auto root = impl();
    if (!root->requires_grad) {
        throw std::logic_error("backward() requires a tensor with requires_grad=true");
    }
    if (root->graph_freed) {
        throw std::logic_error("the computation graph has already been freed; pass "
                               "retain_graph=true on the first backward call");
    }

    Tensor root_gradient;
    if (gradient.has_value()) {
        if (gradient->shape() != shape()) {
            throw std::invalid_argument("backward gradient shape " +
                                        shape_string(gradient->shape()) +
                                        " does not match output shape " + shape_string(shape()));
        }
        root_gradient = gradient->device() == device() ? *gradient : gradient->to(device());
    } else {
        if (!is_scalar()) {
            throw std::logic_error("a gradient is required for non-scalar outputs");
        }
        root_gradient = Tensor::scalar(1.0F, false, {}, device());
    }

    std::vector<std::shared_ptr<detail::TensorImpl>> topological_order;
    std::unordered_set<detail::TensorImpl*> visited;
    std::function<void(const std::shared_ptr<detail::TensorImpl>&)> visit =
        [&](const std::shared_ptr<detail::TensorImpl>& implementation) {
            if (!visited.insert(implementation.get()).second) {
                return;
            }
            if (implementation->grad_fn != nullptr) {
                for (const auto& parent : implementation->grad_fn->parents) {
                    visit(parent);
                }
            }
            topological_order.push_back(implementation);
        };
    visit(root);

    for (const auto& implementation : topological_order) {
        if (implementation->grad_fn != nullptr) {
            implementation->gradient.clear();
            implementation->gradient_buffer.reset();
        }
    }
    detail_accumulate(*this, root_gradient);

    for (auto iterator = topological_order.rbegin(); iterator != topological_order.rend();
         ++iterator) {
        const auto& implementation = *iterator;
        const auto node = implementation->grad_fn;
        if (node == nullptr ||
            (implementation->gradient.empty() && implementation->gradient_buffer == nullptr)) {
            continue;
        }
        for (std::size_t index = 0; index < node->parents.size(); ++index) {
            if (node->parents[index]->version != node->parent_versions[index]) {
                throw std::logic_error("tensor required by '" + node->operation +
                                       "' was modified after the forward pass");
            }
        }
        std::shared_ptr<detail::TensorImpl> upstream_implementation =
            std::make_shared<detail::TensorImpl>();
        upstream_implementation->shape = implementation->shape;
        upstream_implementation->device = implementation->device;
        if (implementation->gradient_buffer != nullptr) {
            upstream_implementation->opencl_buffer = implementation->gradient_buffer;
        } else {
            upstream_implementation->data = implementation->gradient;
        }
        node->backward(Tensor(std::move(upstream_implementation)));
    }

    if (!retain_graph) {
        for (const auto& implementation : topological_order) {
            if (implementation->grad_fn != nullptr) {
                implementation->grad_fn.reset();
                implementation->graph_freed = true;
            }
        }
    }
}

NoGradGuard::NoGradGuard() : previous_(gradient_recording_enabled) {
    gradient_recording_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    gradient_recording_enabled = previous_;
}

detail::GradRecordingGuard::GradRecordingGuard() : previous_(gradient_recording_enabled) {
    gradient_recording_enabled = true;
}

detail::GradRecordingGuard::~GradRecordingGuard() {
    gradient_recording_enabled = previous_;
}

bool grad_enabled() noexcept {
    return gradient_recording_enabled;
}

} // namespace clnn
