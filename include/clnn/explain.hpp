#pragma once

#include "clnn/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace clnn::explain {

struct Target final {
    Target(std::size_t target_index, std::optional<std::size_t> target_batch_index = std::nullopt)
        : index(target_index), batch_index(target_batch_index) {}

    std::size_t index;
    std::optional<std::size_t> batch_index;
};

[[nodiscard]] Tensor saliency(nn::Module& model, const Tensor& input, Target target);
[[nodiscard]] Tensor input_x_gradient(nn::Module& model, const Tensor& input, Target target);
[[nodiscard]] Tensor integrated_gradients(nn::Module& model, const Tensor& input, Target target,
                                          const Tensor& baseline, std::size_t steps = 50);
[[nodiscard]] Tensor smoothgrad(nn::Module& model, const Tensor& input, Target target,
                                std::size_t samples = 50, float noise_stddev = 0.1F,
                                std::uint64_t seed = 0);

} // namespace clnn::explain
