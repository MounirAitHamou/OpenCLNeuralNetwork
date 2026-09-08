#pragma once

#include "clnn/autograd/tensor.hpp"

#include <cstddef>
#include <string>

namespace clnn {

struct TensorStatistics final {
    float minimum;
    float maximum;
    float mean;
    float standard_deviation;
    float l2_norm;
    std::size_t non_finite;
};

[[nodiscard]] TensorStatistics statistics(const Tensor& tensor);
[[nodiscard]] std::string inspect(const Tensor& tensor, std::size_t maximum_values = 12);

} // namespace clnn
