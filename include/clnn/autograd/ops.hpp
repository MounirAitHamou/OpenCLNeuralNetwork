#pragma once

#include "clnn/autograd/tensor.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace clnn {

[[nodiscard]] Tensor add(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor subtract(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor multiply(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor divide(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor negate(const Tensor& input);
[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor conv2d(const Tensor& input, const Tensor& weight,
                            const std::optional<Tensor>& bias = std::nullopt,
                            std::array<std::size_t, 2> stride = {1, 1},
                            std::array<std::size_t, 2> padding = {0, 0});
[[nodiscard]] Tensor max_pool2d(const Tensor& input, std::array<std::size_t, 2> kernel_size,
                                std::array<std::size_t, 2> stride = {0, 0},
                                std::array<std::size_t, 2> padding = {0, 0});
[[nodiscard]] Tensor avg_pool2d(const Tensor& input, std::array<std::size_t, 2> kernel_size,
                                std::array<std::size_t, 2> stride = {0, 0},
                                std::array<std::size_t, 2> padding = {0, 0});
[[nodiscard]] Tensor sum(const Tensor& input);
[[nodiscard]] Tensor sum(const Tensor& input, std::size_t dimension, bool keep_dimensions = false);
[[nodiscard]] Tensor mean(const Tensor& input);
[[nodiscard]] Tensor mean(const Tensor& input, std::size_t dimension, bool keep_dimensions = false);
[[nodiscard]] Tensor pow(const Tensor& input, float exponent);
[[nodiscard]] Tensor exp(const Tensor& input);
[[nodiscard]] Tensor log(const Tensor& input);
[[nodiscard]] Tensor relu(const Tensor& input);
[[nodiscard]] Tensor leaky_relu(const Tensor& input, float negative_slope = 0.01F);
[[nodiscard]] Tensor sigmoid(const Tensor& input);
[[nodiscard]] Tensor tanh(const Tensor& input);
[[nodiscard]] Tensor softmax(const Tensor& input, std::size_t dimension);
[[nodiscard]] Tensor reshape(const Tensor& input, Shape shape);
[[nodiscard]] Tensor transpose(const Tensor& input);

[[nodiscard]] Tensor mse_loss(const Tensor& prediction, const Tensor& target);
[[nodiscard]] Tensor binary_cross_entropy(const Tensor& prediction, const Tensor& target,
                                          float epsilon = 1.0e-7F);
[[nodiscard]] Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target);
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const std::vector<std::size_t>& labels);

[[nodiscard]] Tensor operator+(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator*(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator/(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& input);
[[nodiscard]] Tensor operator+(const Tensor& lhs, float rhs);
[[nodiscard]] Tensor operator+(float lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& lhs, float rhs);
[[nodiscard]] Tensor operator-(float lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator*(const Tensor& lhs, float rhs);
[[nodiscard]] Tensor operator*(float lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator/(const Tensor& lhs, float rhs);
[[nodiscard]] Tensor operator/(float lhs, const Tensor& rhs);

} // namespace clnn
