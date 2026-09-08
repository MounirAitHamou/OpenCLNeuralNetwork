#include "clnn/explain.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clnn::explain {
namespace {

class EvaluationGuard final {
  public:
    explicit EvaluationGuard(nn::Module& model) : model_(model), was_training_(model.training()) {
        model_.eval();
    }
    ~EvaluationGuard() {
        try {
            model_.train(was_training_);
        } catch (...) {
        }
    }
    EvaluationGuard(const EvaluationGuard&) = delete;
    EvaluationGuard& operator=(const EvaluationGuard&) = delete;

  private:
    nn::Module& model_;
    bool was_training_;
};

Tensor clean_copy(const Tensor& tensor) {
    return detail_clone_leaf(tensor, false);
}

Tensor copy_to_input_device(const Tensor& tensor, const Device& device) {
    if (tensor.device() == device)
        return clean_copy(tensor);
    return Tensor(tensor.data(), tensor.shape(), false, tensor.name(), device);
}

Tensor target_gradient(const Tensor& output, const Target target) {
    std::vector<float> values(output.size(), 0.0F);
    if (output.ndim() == 0) {
        if (target.index != 0 || target.batch_index.has_value())
            throw std::out_of_range(
                "scalar model output only supports target index 0 without a batch index");
        values[0] = 1.0F;
    } else if (output.ndim() == 1) {
        if (target.batch_index.has_value())
            throw std::invalid_argument("a batch index requires model output rank 2 or greater");
        if (target.index >= output.size())
            throw std::out_of_range("target index is out of range for model output " +
                                    shape_string(output.shape()));
        values[target.index] = 1.0F;
    } else {
        const auto batch = output.shape().front();
        const auto outputs_per_sample = output.size() / batch;
        if (target.index >= outputs_per_sample)
            throw std::out_of_range("target index is out of range for each model output sample " +
                                    shape_string(output.shape()));
        if (target.batch_index.has_value()) {
            if (*target.batch_index >= batch)
                throw std::out_of_range("target batch index is out of range for model output " +
                                        shape_string(output.shape()));
            values[*target.batch_index * outputs_per_sample + target.index] = 1.0F;
        } else {
            for (std::size_t batch_index = 0; batch_index < batch; ++batch_index)
                values[batch_index * outputs_per_sample + target.index] = 1.0F;
        }
    }
    return Tensor(std::move(values), output.shape(), false, {}, output.device());
}

Tensor input_gradient(nn::Module& model, const Tensor& point, const Target target) {
    auto leaf = detail_clone_leaf(point, true, "explanation_input");
    auto output = model(leaf);
    auto selector = target_gradient(output, target);
    if (!output.requires_grad())
        throw std::logic_error("model output is not differentiable with respect to the input");
    detail_backward_to(std::move(output), leaf, std::move(selector));
    if (!leaf.has_grad())
        throw std::logic_error("selected model output does not depend on the input");
    return detail_gradient_tensor(leaf);
}

Tensor magnitude(const Tensor& input) {
    // Reuses backend-native operations, including their OpenCL kernels, without a host read.
    return relu(input) + relu(-input);
}

std::mt19937 seeded_generator(const std::uint64_t seed) {
    std::seed_seq sequence{static_cast<std::uint32_t>(seed),
                           static_cast<std::uint32_t>(seed >> 32U)};
    return std::mt19937(sequence);
}

} // namespace

Tensor saliency(nn::Module& model, const Tensor& input, const Target target) {
    detail::GradRecordingGuard gradients;
    EvaluationGuard evaluation(model);
    const auto point = clean_copy(input);
    return magnitude(input_gradient(model, point, target));
}

Tensor input_x_gradient(nn::Module& model, const Tensor& input, const Target target) {
    detail::GradRecordingGuard gradients;
    EvaluationGuard evaluation(model);
    const auto point = clean_copy(input);
    return point * input_gradient(model, point, target);
}

Tensor integrated_gradients(nn::Module& model, const Tensor& input, const Target target,
                            const Tensor& baseline, const std::size_t steps) {
    if (steps == 0)
        throw std::invalid_argument("integrated gradients steps must be greater than zero");
    if (baseline.shape() != input.shape())
        throw std::invalid_argument("integrated gradients baseline shape " +
                                    shape_string(baseline.shape()) +
                                    " does not match input shape " +
                                    shape_string(input.shape()));

    detail::GradRecordingGuard gradients;
    EvaluationGuard evaluation(model);
    const auto point = clean_copy(input);
    const auto base = copy_to_input_device(baseline, input.device());
    const auto delta = point - base;
    auto gradient_sum = Tensor::zeros(input.shape(), false, {}, input.device());
    for (std::size_t step = 1; step <= steps; ++step) {
        const auto alpha = static_cast<float>(step) / static_cast<float>(steps);
        const auto sample = base + delta * alpha;
        gradient_sum = gradient_sum + input_gradient(model, sample, target);
    }
    return delta * (gradient_sum / static_cast<float>(steps));
}

Tensor smoothgrad(nn::Module& model, const Tensor& input, const Target target,
                  const std::size_t samples, const float noise_stddev, const std::uint64_t seed) {
    if (samples == 0)
        throw std::invalid_argument("SmoothGrad samples must be greater than zero");
    if (!std::isfinite(noise_stddev) || noise_stddev < 0.0F)
        throw std::invalid_argument(
            "SmoothGrad noise standard deviation must be finite and non-negative");

    detail::GradRecordingGuard gradients;
    EvaluationGuard evaluation(model);
    const auto point = clean_copy(input);
    auto generator = seeded_generator(seed);
    auto attribution_sum = Tensor::zeros(input.shape(), false, {}, input.device());
    for (std::size_t sample_index = 0; sample_index < samples; ++sample_index) {
        const auto noise = noise_stddev == 0.0F
                               ? Tensor::zeros(input.shape(), false, {}, input.device())
                               : Tensor::randn(input.shape(), generator, 0.0F, noise_stddev, false,
                                               {}, input.device());
        attribution_sum =
            attribution_sum + magnitude(input_gradient(model, point + noise, target));
    }
    return attribution_sum / static_cast<float>(samples);
}

} // namespace clnn::explain
