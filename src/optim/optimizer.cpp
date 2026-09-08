#include "clnn/optim/optimizer.hpp"

#include "opencl/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace clnn::optim {
namespace {

void validate_hyperparameter(const bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

constexpr char optimizer_magic[] = {'C', 'L', 'N', 'N', 'O', 'P', 'T', '2'};
constexpr std::uint32_t parameter_group_magic = 0x47525031U;
template <typename Value> void write_value(std::ostream& stream, const Value& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(Value));
    if (!stream)
        throw std::runtime_error("failed to write optimizer state");
}
template <typename Value> Value read_value(std::istream& stream) {
    Value value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!stream)
        throw std::runtime_error("optimizer state is truncated");
    return value;
}
void write_vectors(std::ostream& stream, const std::vector<std::vector<float>>& vectors) {
    write_value(stream, static_cast<std::uint64_t>(vectors.size()));
    for (const auto& vector : vectors) {
        write_value(stream, static_cast<std::uint64_t>(vector.size()));
        stream.write(reinterpret_cast<const char*>(vector.data()),
                     static_cast<std::streamsize>(vector.size() * sizeof(float)));
    }
}
std::vector<std::vector<float>> read_vectors(std::istream& stream,
                                             const std::vector<Tensor*>& parameters) {
    const auto count = read_value<std::uint64_t>(stream);
    if (count != parameters.size())
        throw std::runtime_error("optimizer parameter count mismatch");
    std::vector<std::vector<float>> result(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto size = read_value<std::uint64_t>(stream);
        if (size != 0 && size != parameters[index]->size())
            throw std::runtime_error("optimizer tensor state size mismatch");
        result[index].resize(static_cast<std::size_t>(size));
        stream.read(reinterpret_cast<char*>(result[index].data()),
                    static_cast<std::streamsize>(result[index].size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("optimizer state is truncated");
    }
    return result;
}

std::vector<std::vector<float>>
state_vectors(const std::vector<Tensor*>& parameters, const std::vector<std::vector<float>>& host,
              const std::vector<std::shared_ptr<opencl::Buffer>>& device) {
    if (host.size() != parameters.size() || device.size() != parameters.size())
        throw std::logic_error("optimizer state count mismatch");
    auto result = host;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (device[index] != nullptr)
            result[index] = device[index]->read();
    }
    return result;
}

void place_state_on_devices(const std::vector<Tensor*>& parameters,
                            std::vector<std::vector<float>>& host,
                            std::vector<std::shared_ptr<opencl::Buffer>>& device) {
    device.assign(parameters.size(), nullptr);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (parameters[index]->device().type() == DeviceType::opencl && !host[index].empty()) {
            device[index] = std::make_shared<opencl::Buffer>(
                parameters[index]->device(), host[index].size(), host[index].data());
            host[index].clear();
        }
    }
}

void write_parameter_groups(std::ostream& stream, const std::vector<ParameterGroup>& groups) {
    write_value(stream, parameter_group_magic);
    write_value(stream, static_cast<std::uint64_t>(groups.size()));
    for (const auto& group : groups) {
        write_value(stream, group.learning_rate);
        write_value(stream, group.weight_decay);
        write_value(stream, static_cast<std::uint64_t>(group.parameters.size()));
    }
}

bool read_parameter_groups(std::istream& stream, std::vector<ParameterGroup>& groups) {
    if (stream.peek() == std::char_traits<char>::eof())
        return false;
    if (read_value<std::uint32_t>(stream) != parameter_group_magic)
        throw std::runtime_error("optimizer parameter-group state is corrupt");
    const auto count = read_value<std::uint64_t>(stream);
    if (count != groups.size())
        throw std::runtime_error("optimizer parameter-group count mismatch");
    for (auto& group : groups) {
        const auto learning_rate = read_value<float>(stream);
        const auto weight_decay = read_value<float>(stream);
        const auto parameter_count = read_value<std::uint64_t>(stream);
        if (learning_rate < 0.0F || weight_decay < 0.0F ||
            parameter_count != group.parameters.size())
            throw std::runtime_error("optimizer parameter-group state mismatch");
        group.learning_rate = learning_rate;
        group.weight_decay = weight_decay;
    }
    return true;
}

} // namespace

Optimizer::Optimizer(std::vector<Tensor*> parameters)
    : Optimizer(std::vector<ParameterGroup>{{std::move(parameters), 1.0F, 0.0F}}) {}

Optimizer::Optimizer(std::vector<ParameterGroup> parameter_groups)
    : parameter_groups_(std::move(parameter_groups)) {
    std::unordered_set<Tensor*> unique;
    for (std::size_t group_index = 0; group_index < parameter_groups_.size(); ++group_index) {
        const auto& group = parameter_groups_[group_index];
        validate_hyperparameter(group.learning_rate > 0.0F,
                                "parameter-group learning rate must be positive");
        validate_hyperparameter(group.weight_decay >= 0.0F,
                                "parameter-group weight decay must be non-negative");
        for (auto* parameter : group.parameters) {
            if (!unique.insert(parameter).second)
                throw std::invalid_argument("an optimizer parameter cannot occur in two groups");
            parameters_.push_back(parameter);
            parameter_group_indices_.push_back(group_index);
        }
    }
    for (const auto* parameter : parameters_) {
        if (parameter == nullptr || !parameter->defined()) {
            throw std::invalid_argument("optimizer parameters must be valid tensors");
        }
        if (!parameter->requires_grad() || !parameter->is_leaf()) {
            throw std::invalid_argument(
                "optimizer parameters must be leaf tensors requiring gradients");
        }
    }
}

void Optimizer::zero_grad() {
    for (auto* parameter : parameters_) {
        parameter->zero_grad();
    }
}

std::size_t Optimizer::parameter_group_count() const noexcept {
    return parameter_groups_.size();
}

float Optimizer::learning_rate(const std::size_t group) const {
    if (group >= parameter_groups_.size())
        throw std::out_of_range("optimizer parameter-group index is out of range");
    return parameter_groups_[group].learning_rate;
}

void Optimizer::set_learning_rate(const float learning_rate, const std::size_t group) {
    validate_hyperparameter(learning_rate >= 0.0F, "learning rate must be non-negative");
    if (group >= parameter_groups_.size())
        throw std::out_of_range("optimizer parameter-group index is out of range");
    parameter_groups_[group].learning_rate = learning_rate;
}

SGD::SGD(std::vector<Tensor*> parameters, const float learning_rate, const float momentum,
         const float weight_decay)
    : Optimizer(std::move(parameters)), learning_rate_(learning_rate), momentum_(momentum),
      weight_decay_(weight_decay), velocity_(parameters_.size()),
      velocity_buffers_(parameters_.size()) {
    validate_hyperparameter(learning_rate > 0.0F, "learning rate must be positive");
    validate_hyperparameter(momentum >= 0.0F && momentum < 1.0F, "momentum must be in [0, 1)");
    validate_hyperparameter(weight_decay >= 0.0F, "weight decay must be non-negative");
    parameter_groups_[0].learning_rate = learning_rate;
    parameter_groups_[0].weight_decay = weight_decay;
}

SGD::SGD(ParameterGroups parameter_groups, const float momentum)
    : Optimizer(std::move(parameter_groups.groups)),
      learning_rate_(parameter_groups_.empty() ? 1.0F : parameter_groups_.front().learning_rate),
      momentum_(momentum),
      weight_decay_(parameter_groups_.empty() ? 0.0F : parameter_groups_.front().weight_decay),
      velocity_(parameters_.size()), velocity_buffers_(parameters_.size()) {
    validate_hyperparameter(momentum >= 0.0F && momentum < 1.0F, "momentum must be in [0, 1)");
}

void SGD::step() {
    NoGradGuard guard;
    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        auto* parameter = parameters_[parameter_index];
        const auto& group = parameter_groups_[parameter_group_indices_[parameter_index]];
        if (!parameter->has_grad()) {
            continue;
        }
        if (parameter->device().type() == DeviceType::opencl) {
            if (velocity_buffers_[parameter_index] != nullptr &&
                velocity_buffers_[parameter_index]->device() != parameter->device()) {
                velocity_[parameter_index] = velocity_buffers_[parameter_index]->read();
                velocity_buffers_[parameter_index].reset();
            }
            if (velocity_buffers_[parameter_index] == nullptr) {
                if (velocity_[parameter_index].empty())
                    velocity_[parameter_index].assign(parameter->size(), 0.0F);
                velocity_buffers_[parameter_index] = std::make_shared<opencl::Buffer>(
                    parameter->device(), parameter->size(), velocity_[parameter_index].data());
                velocity_[parameter_index].clear();
            }
            opencl::sgd_update(*detail_opencl_buffer(*parameter),
                               *detail_opencl_gradient_buffer(*parameter),
                               *velocity_buffers_[parameter_index], group.learning_rate, momentum_,
                               group.weight_decay);
            detail_mark_opencl_modified(*parameter);
            continue;
        }
        if (velocity_[parameter_index].empty() && velocity_buffers_[parameter_index] != nullptr) {
            velocity_[parameter_index] = velocity_buffers_[parameter_index]->read();
            velocity_buffers_[parameter_index].reset();
        }
        const auto& gradient = parameter->grad();
        if (velocity_[parameter_index].empty()) {
            velocity_[parameter_index].assign(parameter->size(), 0.0F);
        }
        auto values = parameter->data();
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto direction = gradient[index] + group.weight_decay * values[index];
            if (momentum_ != 0.0F) {
                velocity_[parameter_index][index] =
                    momentum_ * velocity_[parameter_index][index] + direction;
                direction = velocity_[parameter_index][index];
            }
            values[index] -= group.learning_rate * direction;
        }
        parameter->set_data(std::move(values));
    }
}

Adam::Adam(std::vector<Tensor*> parameters, const float learning_rate, const float beta1,
           const float beta2, const float epsilon, const float weight_decay,
           const bool decoupled_weight_decay)
    : Optimizer(std::move(parameters)), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon), weight_decay_(weight_decay),
      decoupled_weight_decay_(decoupled_weight_decay), first_moment_(parameters_.size()),
      second_moment_(parameters_.size()), first_moment_buffers_(parameters_.size()),
      second_moment_buffers_(parameters_.size()) {
    validate_hyperparameter(learning_rate > 0.0F, "learning rate must be positive");
    validate_hyperparameter(beta1 >= 0.0F && beta1 < 1.0F, "beta1 must be in [0, 1)");
    validate_hyperparameter(beta2 >= 0.0F && beta2 < 1.0F, "beta2 must be in [0, 1)");
    validate_hyperparameter(epsilon > 0.0F, "epsilon must be positive");
    validate_hyperparameter(weight_decay >= 0.0F, "weight decay must be non-negative");
    parameter_groups_[0].learning_rate = learning_rate;
    parameter_groups_[0].weight_decay = weight_decay;
}

Adam::Adam(ParameterGroups parameter_groups, const float beta1, const float beta2,
           const float epsilon, const bool decoupled_weight_decay)
    : Optimizer(std::move(parameter_groups.groups)),
      learning_rate_(parameter_groups_.empty() ? 1.0e-3F : parameter_groups_.front().learning_rate),
      beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(parameter_groups_.empty() ? 0.0F : parameter_groups_.front().weight_decay),
      decoupled_weight_decay_(decoupled_weight_decay), first_moment_(parameters_.size()),
      second_moment_(parameters_.size()), first_moment_buffers_(parameters_.size()),
      second_moment_buffers_(parameters_.size()) {
    validate_hyperparameter(beta1 >= 0.0F && beta1 < 1.0F, "beta1 must be in [0, 1)");
    validate_hyperparameter(beta2 >= 0.0F && beta2 < 1.0F, "beta2 must be in [0, 1)");
    validate_hyperparameter(epsilon > 0.0F, "epsilon must be positive");
}

void Adam::step() {
    ++step_;
    const auto first_correction = 1.0F - std::pow(beta1_, static_cast<float>(step_));
    const auto second_correction = 1.0F - std::pow(beta2_, static_cast<float>(step_));
    NoGradGuard guard;
    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        auto* parameter = parameters_[parameter_index];
        const auto& group = parameter_groups_[parameter_group_indices_[parameter_index]];
        if (!parameter->has_grad()) {
            continue;
        }
        if (parameter->device().type() == DeviceType::opencl) {
            if (first_moment_buffers_[parameter_index] != nullptr &&
                first_moment_buffers_[parameter_index]->device() != parameter->device()) {
                first_moment_[parameter_index] = first_moment_buffers_[parameter_index]->read();
                second_moment_[parameter_index] = second_moment_buffers_[parameter_index]->read();
                first_moment_buffers_[parameter_index].reset();
                second_moment_buffers_[parameter_index].reset();
            }
            if (first_moment_buffers_[parameter_index] == nullptr) {
                if (first_moment_[parameter_index].empty()) {
                    first_moment_[parameter_index].assign(parameter->size(), 0.0F);
                    second_moment_[parameter_index].assign(parameter->size(), 0.0F);
                }
                first_moment_buffers_[parameter_index] = std::make_shared<opencl::Buffer>(
                    parameter->device(), parameter->size(), first_moment_[parameter_index].data());
                second_moment_buffers_[parameter_index] = std::make_shared<opencl::Buffer>(
                    parameter->device(), parameter->size(), second_moment_[parameter_index].data());
                first_moment_[parameter_index].clear();
                second_moment_[parameter_index].clear();
            }
            opencl::adam_update(
                *detail_opencl_buffer(*parameter), *detail_opencl_gradient_buffer(*parameter),
                *first_moment_buffers_[parameter_index], *second_moment_buffers_[parameter_index],
                group.learning_rate, beta1_, beta2_, epsilon_, group.weight_decay, first_correction,
                second_correction, decoupled_weight_decay_);
            detail_mark_opencl_modified(*parameter);
            continue;
        }
        if (first_moment_[parameter_index].empty() &&
            first_moment_buffers_[parameter_index] != nullptr) {
            first_moment_[parameter_index] = first_moment_buffers_[parameter_index]->read();
            second_moment_[parameter_index] = second_moment_buffers_[parameter_index]->read();
            first_moment_buffers_[parameter_index].reset();
            second_moment_buffers_[parameter_index].reset();
        }
        const auto& gradient = parameter->grad();
        auto& first = first_moment_[parameter_index];
        auto& second = second_moment_[parameter_index];
        if (first.empty()) {
            first.assign(parameter->size(), 0.0F);
            second.assign(parameter->size(), 0.0F);
        }
        auto values = parameter->data();
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto direction = gradient[index];
            if (!decoupled_weight_decay_) {
                direction += group.weight_decay * values[index];
            } else {
                values[index] *= 1.0F - group.learning_rate * group.weight_decay;
            }
            first[index] = beta1_ * first[index] + (1.0F - beta1_) * direction;
            second[index] = beta2_ * second[index] + (1.0F - beta2_) * direction * direction;
            const auto corrected_first = first[index] / first_correction;
            const auto corrected_second = second[index] / second_correction;
            values[index] -=
                group.learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon_);
        }
        parameter->set_data(std::move(values));
    }
}

AdamW::AdamW(std::vector<Tensor*> parameters, const float learning_rate, const float beta1,
             const float beta2, const float epsilon, const float weight_decay)
    : Optimizer(parameters), implementation_(std::move(parameters), learning_rate, beta1, beta2,
                                             epsilon, weight_decay, true) {
    parameter_groups_[0].learning_rate = learning_rate;
    parameter_groups_[0].weight_decay = weight_decay;
}

AdamW::AdamW(ParameterGroups parameter_groups, const float beta1, const float beta2,
             const float epsilon)
    : Optimizer(parameter_groups.groups),
      implementation_(ParameterGroups{std::move(parameter_groups.groups)}, beta1, beta2, epsilon,
                      true) {}

void AdamW::step() {
    implementation_.step();
}

void AdamW::set_learning_rate(const float learning_rate, const std::size_t group) {
    Optimizer::set_learning_rate(learning_rate, group);
    implementation_.set_learning_rate(learning_rate, group);
}

LRScheduler::LRScheduler(Optimizer& optimizer) : optimizer_(&optimizer) {
    initial_learning_rates_.reserve(optimizer.parameter_group_count());
    for (std::size_t group = 0; group < optimizer.parameter_group_count(); ++group)
        initial_learning_rates_.push_back(optimizer.learning_rate(group));
}
std::size_t LRScheduler::steps() const noexcept {
    return steps_;
}

ExponentialLR::ExponentialLR(Optimizer& optimizer, const float gamma)
    : LRScheduler(optimizer), gamma_(gamma) {
    validate_hyperparameter(gamma > 0.0F, "scheduler gamma must be positive");
}
void ExponentialLR::step() {
    ++steps_;
    for (std::size_t group = 0; group < optimizer_->parameter_group_count(); ++group)
        optimizer_->set_learning_rate(optimizer_->learning_rate(group) * gamma_, group);
}

StepLR::StepLR(Optimizer& optimizer, const std::size_t step_size, const float gamma)
    : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {
    validate_hyperparameter(step_size > 0, "scheduler step size must be positive");
    validate_hyperparameter(gamma > 0.0F, "scheduler gamma must be positive");
}
void StepLR::step() {
    ++steps_;
    if (steps_ % step_size_ != 0)
        return;
    for (std::size_t group = 0; group < optimizer_->parameter_group_count(); ++group)
        optimizer_->set_learning_rate(optimizer_->learning_rate(group) * gamma_, group);
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, const std::size_t maximum_steps,
                                     const float minimum_learning_rate)
    : LRScheduler(optimizer), maximum_steps_(maximum_steps),
      minimum_learning_rate_(minimum_learning_rate) {
    validate_hyperparameter(maximum_steps > 0, "cosine scheduler maximum steps must be positive");
    validate_hyperparameter(minimum_learning_rate >= 0.0F,
                            "minimum learning rate must be non-negative");
}
void CosineAnnealingLR::step() {
    ++steps_;
    const auto progress =
        static_cast<float>(std::min(steps_, maximum_steps_)) / static_cast<float>(maximum_steps_);
    const auto cosine = (1.0F + std::cos(std::numbers::pi_v<float> * progress)) * 0.5F;
    for (std::size_t group = 0; group < optimizer_->parameter_group_count(); ++group) {
        const auto learning_rate =
            minimum_learning_rate_ +
            (initial_learning_rates_[group] - minimum_learning_rate_) * cosine;
        optimizer_->set_learning_rate(learning_rate, group);
    }
}

void save_state_dict(const Optimizer& optimizer, const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot open optimizer state for writing: " + path.string());
    stream.write(optimizer_magic, sizeof(optimizer_magic));
    write_value(stream, static_cast<std::uint64_t>(optimizer.parameters_.size()));
    for (const auto* parameter : optimizer.parameters_) {
        write_value(stream, static_cast<std::uint64_t>(parameter->ndim()));
        for (const auto dimension : parameter->shape())
            write_value(stream, static_cast<std::uint64_t>(dimension));
    }
    if (const auto* sgd = dynamic_cast<const SGD*>(&optimizer)) {
        write_value(stream, std::uint32_t{1});
        write_value(stream, sgd->learning_rate_);
        write_value(stream, sgd->momentum_);
        write_value(stream, sgd->weight_decay_);
        write_vectors(stream,
                      state_vectors(sgd->parameters_, sgd->velocity_, sgd->velocity_buffers_));
        write_parameter_groups(stream, sgd->parameter_groups_);
        return;
    }
    const Adam* adam = dynamic_cast<const Adam*>(&optimizer);
    std::uint32_t type = 2;
    if (const auto* adamw = dynamic_cast<const AdamW*>(&optimizer)) {
        adam = &adamw->implementation_;
        type = 3;
    }
    if (adam == nullptr)
        throw std::runtime_error("unsupported optimizer type");
    write_value(stream, type);
    write_value(stream, adam->learning_rate_);
    write_value(stream, adam->beta1_);
    write_value(stream, adam->beta2_);
    write_value(stream, adam->epsilon_);
    write_value(stream, adam->weight_decay_);
    write_value(stream, static_cast<std::uint64_t>(adam->step_));
    write_vectors(
        stream, state_vectors(adam->parameters_, adam->first_moment_, adam->first_moment_buffers_));
    write_vectors(stream, state_vectors(adam->parameters_, adam->second_moment_,
                                        adam->second_moment_buffers_));
    write_parameter_groups(stream, optimizer.parameter_groups_);
}

void load_state_dict(Optimizer& optimizer, const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open optimizer state for reading: " + path.string());
    char magic[sizeof(optimizer_magic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || !std::equal(std::begin(magic), std::end(magic), std::begin(optimizer_magic)))
        throw std::runtime_error("file is not a supported CLNN optimizer state");
    const auto count = read_value<std::uint64_t>(stream);
    if (count != optimizer.parameters_.size())
        throw std::runtime_error("optimizer parameter count mismatch");
    for (std::size_t index = 0; index < optimizer.parameters_.size(); ++index) {
        const auto rank = read_value<std::uint64_t>(stream);
        Shape shape(static_cast<std::size_t>(rank));
        for (auto& dimension : shape)
            dimension = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
        if (shape != optimizer.parameters_[index]->shape())
            throw std::runtime_error("optimizer parameter shape mismatch");
    }
    const auto type = read_value<std::uint32_t>(stream);
    if (auto* sgd = dynamic_cast<SGD*>(&optimizer)) {
        if (type != 1)
            throw std::runtime_error("optimizer type mismatch");
        sgd->learning_rate_ = read_value<float>(stream);
        sgd->momentum_ = read_value<float>(stream);
        sgd->weight_decay_ = read_value<float>(stream);
        for (auto& group : sgd->parameter_groups_) {
            group.learning_rate = sgd->learning_rate_;
            group.weight_decay = sgd->weight_decay_;
        }
        sgd->velocity_ = read_vectors(stream, optimizer.parameters_);
        place_state_on_devices(optimizer.parameters_, sgd->velocity_, sgd->velocity_buffers_);
        if (read_parameter_groups(stream, sgd->parameter_groups_)) {
            sgd->learning_rate_ = sgd->parameter_groups_.front().learning_rate;
            sgd->weight_decay_ = sgd->parameter_groups_.front().weight_decay;
        }
        return;
    }
    Adam* adam = dynamic_cast<Adam*>(&optimizer);
    std::uint32_t expected = 2;
    if (auto* adamw = dynamic_cast<AdamW*>(&optimizer)) {
        adam = &adamw->implementation_;
        expected = 3;
    }
    if (adam == nullptr || type != expected)
        throw std::runtime_error("optimizer type mismatch");
    adam->learning_rate_ = read_value<float>(stream);
    adam->beta1_ = read_value<float>(stream);
    adam->beta2_ = read_value<float>(stream);
    adam->epsilon_ = read_value<float>(stream);
    adam->weight_decay_ = read_value<float>(stream);
    for (auto& group : adam->parameter_groups_) {
        group.learning_rate = adam->learning_rate_;
        group.weight_decay = adam->weight_decay_;
    }
    for (auto& group : optimizer.parameter_groups_) {
        group.learning_rate = adam->learning_rate_;
        group.weight_decay = adam->weight_decay_;
    }
    adam->step_ = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
    adam->first_moment_ = read_vectors(stream, optimizer.parameters_);
    adam->second_moment_ = read_vectors(stream, optimizer.parameters_);
    place_state_on_devices(optimizer.parameters_, adam->first_moment_, adam->first_moment_buffers_);
    place_state_on_devices(optimizer.parameters_, adam->second_moment_,
                           adam->second_moment_buffers_);
    if (read_parameter_groups(stream, optimizer.parameter_groups_)) {
        adam->parameter_groups_ = optimizer.parameter_groups_;
        adam->learning_rate_ = adam->parameter_groups_.front().learning_rate;
        adam->weight_decay_ = adam->parameter_groups_.front().weight_decay;
    }
}

} // namespace clnn::optim
