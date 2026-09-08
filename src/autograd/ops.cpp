#include "clnn/autograd/ops.hpp"

#include "opencl/backend.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace clnn {
namespace {

Shape broadcast_shape(const Shape& lhs, const Shape& rhs) {
    const auto rank = std::max(lhs.size(), rhs.size());
    Shape output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto lhs_dimension = offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
        const auto rhs_dimension = offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
        if (lhs_dimension != rhs_dimension && lhs_dimension != 1 && rhs_dimension != 1) {
            throw std::invalid_argument("cannot broadcast shapes " + shape_string(lhs) + " and " +
                                        shape_string(rhs));
        }
        output[rank - 1 - offset] = std::max(lhs_dimension, rhs_dimension);
    }
    return output;
}

std::vector<std::size_t> broadcast_indices(const Shape& output_shape, const Shape& input_shape) {
    std::vector<std::size_t> indices(numel(output_shape), 0);
    if (input_shape.empty()) {
        return indices;
    }
    Shape strides(input_shape.size(), 1);
    for (std::size_t index = input_shape.size() - 1; index > 0; --index) {
        strides[index - 1] = strides[index] * input_shape[index];
    }

    for (std::size_t flat = 0; flat < indices.size(); ++flat) {
        auto remainder = flat;
        std::size_t input_flat = 0;
        for (std::size_t reversed = 0; reversed < output_shape.size(); ++reversed) {
            const auto output_axis = output_shape.size() - 1 - reversed;
            const auto coordinate = remainder % output_shape[output_axis];
            remainder /= output_shape[output_axis];
            if (reversed < input_shape.size()) {
                const auto input_axis = input_shape.size() - 1 - reversed;
                if (input_shape[input_axis] != 1) {
                    input_flat += coordinate * strides[input_axis];
                }
            }
        }
        indices[flat] = input_flat;
    }
    return indices;
}

using Derivative = std::function<float(float, float)>;

Tensor opencl_gradient(std::shared_ptr<opencl::Buffer> buffer, Shape shape, const Device device) {
    return detail_make_opencl_result(std::move(buffer), std::move(shape), {}, {}, {}, device);
}

Tensor binary_operation(const Tensor& lhs, const Tensor& rhs,
                        const std::function<float(float, float)>& forward,
                        Derivative lhs_derivative, Derivative rhs_derivative, std::string operation,
                        const opencl::BinaryOperation gpu_operation) {
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument("binary operation tensors must be on the same device");
    }
    const auto output_shape = broadcast_shape(lhs.shape(), rhs.shape());
    const auto lhs_indices = broadcast_indices(output_shape, lhs.shape());
    const auto rhs_indices = broadcast_indices(output_shape, rhs.shape());
    auto backward = [lhs, rhs, lhs_indices, rhs_indices, lhs_derivative = std::move(lhs_derivative),
                     rhs_derivative = std::move(rhs_derivative),
                     gpu_operation](const Tensor& upstream) {
        if (lhs.device().type() == DeviceType::opencl) {
            if (lhs.requires_grad()) {
                detail_accumulate(
                    lhs, opencl_gradient(opencl::binary_gradient(
                                             gpu_operation, true, *detail_opencl_buffer(lhs),
                                             *detail_opencl_buffer(rhs), lhs_indices, rhs_indices,
                                             *detail_opencl_buffer(upstream), lhs.size()),
                                         lhs.shape(), lhs.device()));
            }
            if (rhs.requires_grad()) {
                detail_accumulate(
                    rhs, opencl_gradient(opencl::binary_gradient(
                                             gpu_operation, false, *detail_opencl_buffer(lhs),
                                             *detail_opencl_buffer(rhs), lhs_indices, rhs_indices,
                                             *detail_opencl_buffer(upstream), rhs.size()),
                                         rhs.shape(), rhs.device()));
            }
            return;
        }
        const auto& upstream_data = upstream.data();
        const auto& lhs_data = lhs.data();
        const auto& rhs_data = rhs.data();
        if (lhs.requires_grad()) {
            std::vector<float> gradient(lhs.size(), 0.0F);
            for (std::size_t index = 0; index < upstream_data.size(); ++index) {
                gradient[lhs_indices[index]] +=
                    upstream_data[index] *
                    lhs_derivative(lhs_data[lhs_indices[index]], rhs_data[rhs_indices[index]]);
            }
            detail_accumulate(lhs, gradient);
        }
        if (rhs.requires_grad()) {
            std::vector<float> gradient(rhs.size(), 0.0F);
            for (std::size_t index = 0; index < upstream_data.size(); ++index) {
                gradient[rhs_indices[index]] +=
                    upstream_data[index] *
                    rhs_derivative(lhs_data[lhs_indices[index]], rhs_data[rhs_indices[index]]);
            }
            detail_accumulate(rhs, gradient);
        }
    };
    if (lhs.device().type() == DeviceType::opencl) {
        auto output = opencl::binary(gpu_operation, *detail_opencl_buffer(lhs),
                                     *detail_opencl_buffer(rhs), lhs_indices, rhs_indices);
        return detail_make_opencl_result(std::move(output), output_shape, {lhs, rhs},
                                         std::move(backward), std::move(operation), lhs.device());
    }
    std::vector<float> output(lhs_indices.size());
    const auto& lhs_data = lhs.data();
    const auto& rhs_data = rhs.data();
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = forward(lhs_data[lhs_indices[index]], rhs_data[rhs_indices[index]]);
    }
    return detail_make_result(std::move(output), output_shape, {lhs, rhs}, std::move(backward),
                              std::move(operation));
}

void require_same_shape(const Tensor& lhs, const Tensor& rhs, const std::string& operation) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument(operation + " requires equal shapes, got " +
                                    shape_string(lhs.shape()) + " and " +
                                    shape_string(rhs.shape()));
    }
}

} // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs, rhs, std::plus<>(), [](float, float) { return 1.0F; },
        [](float, float) { return 1.0F; }, "add", opencl::BinaryOperation::add);
}

Tensor subtract(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs, rhs, std::minus<>(), [](float, float) { return 1.0F; },
        [](float, float) { return -1.0F; }, "subtract", opencl::BinaryOperation::subtract);
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs, rhs, std::multiplies<>(), [](float, float right) { return right; },
        [](float left, float) { return left; }, "multiply", opencl::BinaryOperation::multiply);
}

Tensor divide(const Tensor& lhs, const Tensor& rhs) {
    return binary_operation(
        lhs, rhs, std::divides<>(), [](float, float right) { return 1.0F / right; },
        [](float left, float right) { return -left / (right * right); }, "divide",
        opencl::BinaryOperation::divide);
}

Tensor negate(const Tensor& input) {
    return input * -1.0F;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument("matmul tensors must be on the same device");
    }
    if (lhs.ndim() != 2 || rhs.ndim() != 2) {
        throw std::invalid_argument("matmul currently requires two rank-2 tensors");
    }
    const auto rows = lhs.shape()[0];
    const auto inner = lhs.shape()[1];
    const auto columns = rhs.shape()[1];
    if (inner != rhs.shape()[0]) {
        throw std::invalid_argument("matmul shape mismatch: " + shape_string(lhs.shape()) +
                                    " and " + shape_string(rhs.shape()));
    }
    std::vector<float> output;
    std::shared_ptr<opencl::Buffer> gpu_output;
    if (lhs.device().type() == DeviceType::opencl) {
        gpu_output = opencl::matrix_multiply(*detail_opencl_buffer(lhs), *detail_opencl_buffer(rhs),
                                             rows, inner, columns);
    } else {
        output.assign(rows * columns, 0.0F);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                for (std::size_t index = 0; index < inner; ++index) {
                    output[row * columns + column] +=
                        lhs.data()[row * inner + index] * rhs.data()[index * columns + column];
                }
            }
        }
    }
    auto backward = [lhs, rhs, rows, inner, columns](const Tensor& upstream) {
        if (lhs.device().type() == DeviceType::opencl) {
            if (lhs.requires_grad()) {
                detail_accumulate(
                    lhs, opencl_gradient(opencl::matmul_lhs_gradient(
                                             *detail_opencl_buffer(rhs),
                                             *detail_opencl_buffer(upstream), rows, inner, columns),
                                         lhs.shape(), lhs.device()));
            }
            if (rhs.requires_grad()) {
                detail_accumulate(
                    rhs, opencl_gradient(opencl::matmul_rhs_gradient(
                                             *detail_opencl_buffer(lhs),
                                             *detail_opencl_buffer(upstream), rows, inner, columns),
                                         rhs.shape(), rhs.device()));
            }
            return;
        }
        const auto& upstream_data = upstream.data();
        const auto& lhs_data = lhs.data();
        const auto& rhs_data = rhs.data();
        if (lhs.requires_grad()) {
            std::vector<float> gradient(lhs.size(), 0.0F);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t index = 0; index < inner; ++index) {
                    for (std::size_t column = 0; column < columns; ++column) {
                        gradient[row * inner + index] += upstream_data[row * columns + column] *
                                                         rhs_data[index * columns + column];
                    }
                }
            }
            detail_accumulate(lhs, gradient);
        }
        if (rhs.requires_grad()) {
            std::vector<float> gradient(rhs.size(), 0.0F);
            for (std::size_t index = 0; index < inner; ++index) {
                for (std::size_t column = 0; column < columns; ++column) {
                    for (std::size_t row = 0; row < rows; ++row) {
                        gradient[index * columns + column] +=
                            lhs_data[row * inner + index] * upstream_data[row * columns + column];
                    }
                }
            }
            detail_accumulate(rhs, gradient);
        }
    };
    if (gpu_output != nullptr) {
        return detail_make_opencl_result(std::move(gpu_output), {rows, columns}, {lhs, rhs},
                                         std::move(backward), "matmul", lhs.device());
    }
    return detail_make_result(std::move(output), {rows, columns}, {lhs, rhs}, std::move(backward),
                              "matmul");
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias,
              const std::array<std::size_t, 2> stride, const std::array<std::size_t, 2> padding) {
    if (input.device() != weight.device() ||
        (bias.has_value() && bias->device() != input.device())) {
        throw std::invalid_argument("conv2d tensors must be on the same device");
    }
    if (input.ndim() != 4 || weight.ndim() != 4) {
        throw std::invalid_argument("conv2d expects input [N, C, H, W] and weight [O, C, KH, KW]");
    }
    if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument("conv2d stride must be non-zero");
    }
    const auto batch = input.shape()[0];
    const auto input_channels = input.shape()[1];
    const auto input_height = input.shape()[2];
    const auto input_width = input.shape()[3];
    const auto output_channels = weight.shape()[0];
    const auto kernel_height = weight.shape()[2];
    const auto kernel_width = weight.shape()[3];
    if (weight.shape()[1] != input_channels) {
        throw std::invalid_argument("conv2d input and weight channel counts differ");
    }
    if (input_height + 2 * padding[0] < kernel_height ||
        input_width + 2 * padding[1] < kernel_width) {
        throw std::invalid_argument("conv2d kernel is larger than the padded input");
    }
    if (bias.has_value() && bias->shape() != Shape{output_channels}) {
        throw std::invalid_argument("conv2d bias must have shape [output_channels]");
    }
    const auto output_height = (input_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
    const auto output_width = (input_width + 2 * padding[1] - kernel_width) / stride[1] + 1;
    const auto output_index = [=](std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
        return ((n * output_channels + c) * output_height + h) * output_width + w;
    };
    const auto input_index = [=](std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
        return ((n * input_channels + c) * input_height + h) * input_width + w;
    };
    const auto weight_index = [=](std::size_t o, std::size_t c, std::size_t h, std::size_t w) {
        return ((o * input_channels + c) * kernel_height + h) * kernel_width + w;
    };

    std::vector<float> output;
    std::shared_ptr<opencl::Buffer> gpu_output;
    if (input.device().type() == DeviceType::opencl) {
        const auto bias_buffer = bias.has_value() ? detail_opencl_buffer(*bias) : nullptr;
        gpu_output = opencl::convolution_2d(
            *detail_opencl_buffer(input), *detail_opencl_buffer(weight), bias_buffer.get(), batch,
            input_channels, input_height, input_width, output_channels, kernel_height, kernel_width,
            output_height, output_width, stride[0], stride[1], padding[0], padding[1]);
    } else {
        output.assign(batch * output_channels * output_height * output_width, 0.0F);
        const auto& input_data = input.data();
        const auto& weight_data = weight.data();
        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t out_channel = 0; out_channel < output_channels; ++out_channel) {
                for (std::size_t out_y = 0; out_y < output_height; ++out_y) {
                    for (std::size_t out_x = 0; out_x < output_width; ++out_x) {
                        float value = bias.has_value() ? bias->data()[out_channel] : 0.0F;
                        for (std::size_t in_channel = 0; in_channel < input_channels;
                             ++in_channel) {
                            for (std::size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                                const auto padded_y = out_y * stride[0] + kernel_y;
                                if (padded_y < padding[0] ||
                                    padded_y - padding[0] >= input_height) {
                                    continue;
                                }
                                const auto in_y = padded_y - padding[0];
                                for (std::size_t kernel_x = 0; kernel_x < kernel_width;
                                     ++kernel_x) {
                                    const auto padded_x = out_x * stride[1] + kernel_x;
                                    if (padded_x < padding[1] ||
                                        padded_x - padding[1] >= input_width) {
                                        continue;
                                    }
                                    const auto in_x = padded_x - padding[1];
                                    value += input_data[input_index(n, in_channel, in_y, in_x)] *
                                             weight_data[weight_index(out_channel, in_channel,
                                                                      kernel_y, kernel_x)];
                                }
                            }
                        }
                        output[output_index(n, out_channel, out_y, out_x)] = value;
                    }
                }
            }
        }
    }

    std::vector<Tensor> parents{input, weight};
    if (bias.has_value()) {
        parents.push_back(*bias);
    }
    auto backward = [=](const Tensor& upstream) {
        if (input.device().type() == DeviceType::opencl) {
            auto gradients = opencl::convolution_2d_gradients(
                *detail_opencl_buffer(input), *detail_opencl_buffer(weight),
                *detail_opencl_buffer(upstream), input.requires_grad(), weight.requires_grad(),
                bias.has_value() && bias->requires_grad(), batch, input_channels, input_height,
                input_width, output_channels, kernel_height, kernel_width, output_height,
                output_width, stride[0], stride[1], padding[0], padding[1]);
            if (input.requires_grad())
                detail_accumulate(input, opencl_gradient(std::move(gradients.input), input.shape(),
                                                         input.device()));
            if (weight.requires_grad())
                detail_accumulate(weight, opencl_gradient(std::move(gradients.weight),
                                                          weight.shape(), weight.device()));
            if (bias.has_value() && bias->requires_grad())
                detail_accumulate(*bias, opencl_gradient(std::move(gradients.bias), bias->shape(),
                                                         bias->device()));
            return;
        }
        const auto& upstream_data = upstream.data();
        const auto& input_data = input.data();
        const auto& weight_data = weight.data();
        std::vector<float> input_gradient;
        std::vector<float> weight_gradient;
        std::vector<float> bias_gradient;
        if (input.requires_grad())
            input_gradient.assign(input.size(), 0.0F);
        if (weight.requires_grad())
            weight_gradient.assign(weight.size(), 0.0F);
        if (bias.has_value() && bias->requires_grad())
            bias_gradient.assign(output_channels, 0.0F);
        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t out_channel = 0; out_channel < output_channels; ++out_channel) {
                for (std::size_t out_y = 0; out_y < output_height; ++out_y) {
                    for (std::size_t out_x = 0; out_x < output_width; ++out_x) {
                        const auto upstream_value =
                            upstream_data[output_index(n, out_channel, out_y, out_x)];
                        if (!bias_gradient.empty())
                            bias_gradient[out_channel] += upstream_value;
                        for (std::size_t in_channel = 0; in_channel < input_channels;
                             ++in_channel) {
                            for (std::size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                                const auto padded_y = out_y * stride[0] + kernel_y;
                                if (padded_y < padding[0] || padded_y - padding[0] >= input_height)
                                    continue;
                                const auto in_y = padded_y - padding[0];
                                for (std::size_t kernel_x = 0; kernel_x < kernel_width;
                                     ++kernel_x) {
                                    const auto padded_x = out_x * stride[1] + kernel_x;
                                    if (padded_x < padding[1] ||
                                        padded_x - padding[1] >= input_width)
                                        continue;
                                    const auto in_x = padded_x - padding[1];
                                    const auto in_offset = input_index(n, in_channel, in_y, in_x);
                                    const auto weight_offset =
                                        weight_index(out_channel, in_channel, kernel_y, kernel_x);
                                    if (!input_gradient.empty()) {
                                        input_gradient[in_offset] +=
                                            upstream_value * weight_data[weight_offset];
                                    }
                                    if (!weight_gradient.empty()) {
                                        weight_gradient[weight_offset] +=
                                            upstream_value * input_data[in_offset];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!input_gradient.empty())
            detail_accumulate(input, input_gradient);
        if (!weight_gradient.empty())
            detail_accumulate(weight, weight_gradient);
        if (!bias_gradient.empty())
            detail_accumulate(*bias, bias_gradient);
    };
    const Shape output_shape{batch, output_channels, output_height, output_width};
    if (gpu_output != nullptr) {
        return detail_make_opencl_result(std::move(gpu_output), output_shape, std::move(parents),
                                         std::move(backward), "conv2d", input.device());
    }
    return detail_make_result(std::move(output), output_shape, std::move(parents),
                              std::move(backward), "conv2d");
}

namespace {

Tensor pool2d_operation(const Tensor& input, const std::array<std::size_t, 2> kernel_size,
                        std::array<std::size_t, 2> stride, const std::array<std::size_t, 2> padding,
                        const opencl::PoolingOperation operation) {
    if (input.ndim() != 4)
        throw std::invalid_argument("pool2d expects input shape [N, C, H, W]");
    if (kernel_size[0] == 0 || kernel_size[1] == 0)
        throw std::invalid_argument("pool2d kernel dimensions must be non-zero");
    if (stride == std::array<std::size_t, 2>{0, 0})
        stride = kernel_size;
    if (stride[0] == 0 || stride[1] == 0)
        throw std::invalid_argument("pool2d stride dimensions must be non-zero");
    if (padding[0] >= kernel_size[0] || padding[1] >= kernel_size[1])
        throw std::invalid_argument("pool2d padding must be smaller than its kernel");
    const auto batch = input.shape()[0];
    const auto channels = input.shape()[1];
    const auto input_height = input.shape()[2];
    const auto input_width = input.shape()[3];
    if (input_height + 2 * padding[0] < kernel_size[0] ||
        input_width + 2 * padding[1] < kernel_size[1])
        throw std::invalid_argument("pool2d kernel is larger than the padded input");
    const auto output_height = (input_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    const auto output_width = (input_width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
    const Shape output_shape{batch, channels, output_height, output_width};

    auto backward = [=](const Tensor& upstream) {
        if (!input.requires_grad())
            return;
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input,
                opencl_gradient(opencl::pool_2d_gradient(
                                    operation, *detail_opencl_buffer(input),
                                    *detail_opencl_buffer(upstream), batch, channels, input_height,
                                    input_width, kernel_size[0], kernel_size[1], output_height,
                                    output_width, stride[0], stride[1], padding[0], padding[1]),
                                input.shape(), input.device()));
            return;
        }
        std::vector<float> gradient(input.size(), 0.0F);
        const auto& values = input.data();
        const auto& upstream_values = upstream.data();
        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
                for (std::size_t oy = 0; oy < output_height; ++oy) {
                    for (std::size_t ox = 0; ox < output_width; ++ox) {
                        std::vector<std::size_t> indices;
                        for (std::size_t ky = 0; ky < kernel_size[0]; ++ky) {
                            const auto padded_y = oy * stride[0] + ky;
                            if (padded_y < padding[0] || padded_y - padding[0] >= input_height)
                                continue;
                            for (std::size_t kx = 0; kx < kernel_size[1]; ++kx) {
                                const auto padded_x = ox * stride[1] + kx;
                                if (padded_x < padding[1] || padded_x - padding[1] >= input_width)
                                    continue;
                                indices.push_back(((n * channels + channel) * input_height +
                                                   padded_y - padding[0]) *
                                                      input_width +
                                                  padded_x - padding[1]);
                            }
                        }
                        const auto output_index =
                            ((n * channels + channel) * output_height + oy) * output_width + ox;
                        if (operation == opencl::PoolingOperation::average) {
                            const auto contribution =
                                upstream_values[output_index] / static_cast<float>(indices.size());
                            for (const auto index : indices)
                                gradient[index] += contribution;
                        } else {
                            auto winner = indices.front();
                            for (const auto index : indices)
                                if (values[index] > values[winner])
                                    winner = index;
                            gradient[winner] += upstream_values[output_index];
                        }
                    }
                }
            }
        }
        detail_accumulate(input, gradient);
    };

    if (input.device().type() == DeviceType::opencl) {
        auto output =
            opencl::pool_2d(operation, *detail_opencl_buffer(input), batch, channels, input_height,
                            input_width, kernel_size[0], kernel_size[1], output_height,
                            output_width, stride[0], stride[1], padding[0], padding[1]);
        return detail_make_opencl_result(std::move(output), output_shape, {input},
                                         std::move(backward), "pool2d", input.device());
    }

    std::vector<float> output(numel(output_shape));
    const auto& values = input.data();
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t oy = 0; oy < output_height; ++oy) {
                for (std::size_t ox = 0; ox < output_width; ++ox) {
                    float value = operation == opencl::PoolingOperation::maximum
                                      ? -std::numeric_limits<float>::infinity()
                                      : 0.0F;
                    std::size_t count = 0;
                    for (std::size_t ky = 0; ky < kernel_size[0]; ++ky) {
                        const auto padded_y = oy * stride[0] + ky;
                        if (padded_y < padding[0] || padded_y - padding[0] >= input_height)
                            continue;
                        for (std::size_t kx = 0; kx < kernel_size[1]; ++kx) {
                            const auto padded_x = ox * stride[1] + kx;
                            if (padded_x < padding[1] || padded_x - padding[1] >= input_width)
                                continue;
                            const auto index =
                                ((n * channels + channel) * input_height + padded_y - padding[0]) *
                                    input_width +
                                padded_x - padding[1];
                            value = operation == opencl::PoolingOperation::maximum
                                        ? std::max(value, values[index])
                                        : value + values[index];
                            ++count;
                        }
                    }
                    if (operation == opencl::PoolingOperation::average)
                        value /= static_cast<float>(count);
                    output[((n * channels + channel) * output_height + oy) * output_width + ox] =
                        value;
                }
            }
        }
    }
    return detail_make_result(std::move(output), output_shape, {input}, std::move(backward),
                              "pool2d");
}

} // namespace

Tensor max_pool2d(const Tensor& input, const std::array<std::size_t, 2> kernel_size,
                  const std::array<std::size_t, 2> stride,
                  const std::array<std::size_t, 2> padding) {
    return pool2d_operation(input, kernel_size, stride, padding, opencl::PoolingOperation::maximum);
}

Tensor avg_pool2d(const Tensor& input, const std::array<std::size_t, 2> kernel_size,
                  const std::array<std::size_t, 2> stride,
                  const std::array<std::size_t, 2> padding) {
    return pool2d_operation(input, kernel_size, stride, padding, opencl::PoolingOperation::average);
}

Tensor sum(const Tensor& input) {
    if (input.device().type() == DeviceType::opencl) {
        auto output = opencl::reduce_sum(*detail_opencl_buffer(input));
        return detail_make_opencl_result(
            std::move(output), {}, {input},
            [input](const Tensor& upstream) {
                if (input.requires_grad()) {
                    detail_accumulate(
                        input, opencl_gradient(opencl::sum_gradient(*detail_opencl_buffer(upstream),
                                                                    input.size()),
                                               input.shape(), input.device()));
                }
            },
            "sum", input.device());
    }
    float value = 0.0F;
    for (const auto element : input.data()) {
        value += element;
    }
    return detail_make_result(
        {value}, {}, {input},
        [input](const Tensor& upstream) {
            if (input.requires_grad()) {
                detail_accumulate(input, std::vector<float>(input.size(), upstream.data().front()));
            }
        },
        "sum");
}

Tensor sum(const Tensor& input, const std::size_t dimension, const bool keep_dimensions) {
    if (dimension >= input.ndim())
        throw std::invalid_argument("sum dimension is out of range");
    std::size_t outer = 1;
    for (std::size_t index = 0; index < dimension; ++index)
        outer *= input.shape()[index];
    const auto axis = input.shape()[dimension];
    std::size_t inner = 1;
    for (std::size_t index = dimension + 1; index < input.ndim(); ++index)
        inner *= input.shape()[index];
    auto output_shape = input.shape();
    if (keep_dimensions)
        output_shape[dimension] = 1;
    else
        output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(dimension));

    auto backward = [input, outer, axis, inner](const Tensor& upstream) {
        if (!input.requires_grad())
            return;
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input, opencl_gradient(opencl::axis_sum_gradient(*detail_opencl_buffer(upstream),
                                                                 outer, axis, inner),
                                       input.shape(), input.device()));
            return;
        }
        std::vector<float> gradient(input.size());
        for (std::size_t outer_index = 0; outer_index < outer; ++outer_index)
            for (std::size_t axis_index = 0; axis_index < axis; ++axis_index)
                for (std::size_t inner_index = 0; inner_index < inner; ++inner_index)
                    gradient[(outer_index * axis + axis_index) * inner + inner_index] =
                        upstream.data()[outer_index * inner + inner_index];
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::reduce_axis_sum(*detail_opencl_buffer(input), outer, axis, inner),
            std::move(output_shape), {input}, std::move(backward), "axis_sum", input.device());
    }
    std::vector<float> output(outer * inner, 0.0F);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index)
        for (std::size_t axis_index = 0; axis_index < axis; ++axis_index)
            for (std::size_t inner_index = 0; inner_index < inner; ++inner_index)
                output[outer_index * inner + inner_index] +=
                    input.data()[(outer_index * axis + axis_index) * inner + inner_index];
    return detail_make_result(std::move(output), std::move(output_shape), {input},
                              std::move(backward), "axis_sum");
}

Tensor mean(const Tensor& input) {
    return sum(input) / static_cast<float>(input.size());
}

Tensor mean(const Tensor& input, const std::size_t dimension, const bool keep_dimensions) {
    if (dimension >= input.ndim())
        throw std::invalid_argument("mean dimension is out of range");
    return sum(input, dimension, keep_dimensions) / static_cast<float>(input.shape()[dimension]);
}

Tensor pow(const Tensor& input, const float exponent) {
    auto backward = [input, exponent](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(input, opencl_gradient(opencl::unary_gradient(
                                                         opencl::UnaryOperation::power,
                                                         *detail_opencl_buffer(input),
                                                         *detail_opencl_buffer(upstream), exponent),
                                                     input.shape(), input.device()));
            return;
        }
        const auto& input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            gradient[index] =
                upstream.data()[index] * exponent * std::pow(input_data[index], exponent - 1.0F);
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::power, *detail_opencl_buffer(input), exponent),
            input.shape(), {input}, std::move(backward), "pow", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(),
                   [exponent](float value) { return std::pow(value, exponent); });
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "pow");
}

Tensor exp(const Tensor& input) {
    auto backward = [input](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input, opencl_gradient(opencl::unary_gradient(opencl::UnaryOperation::exponential,
                                                              *detail_opencl_buffer(input),
                                                              *detail_opencl_buffer(upstream)),
                                       input.shape(), input.device()));
            return;
        }
        const auto& input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            gradient[index] = upstream.data()[index] * std::exp(input_data[index]);
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::exponential, *detail_opencl_buffer(input)),
            input.shape(), {input}, std::move(backward), "exp", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(),
                   [](float value) { return std::exp(value); });
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "exp");
}

Tensor log(const Tensor& input) {
    const auto& input_data = input.data();
    if (std::any_of(input_data.begin(), input_data.end(),
                    [](float value) { return !(value > 0.0F); })) {
        throw std::domain_error("log requires strictly positive values");
    }
    auto backward = [input](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input, opencl_gradient(opencl::unary_gradient(opencl::UnaryOperation::logarithm,
                                                              *detail_opencl_buffer(input),
                                                              *detail_opencl_buffer(upstream)),
                                       input.shape(), input.device()));
            return;
        }
        const auto& backward_input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            gradient[index] = upstream.data()[index] / backward_input_data[index];
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::logarithm, *detail_opencl_buffer(input)),
            input.shape(), {input}, std::move(backward), "log", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(),
                   [](float value) { return std::log(value); });
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "log");
}

Tensor relu(const Tensor& input) {
    auto backward = [input](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input, opencl_gradient(opencl::unary_gradient(opencl::UnaryOperation::relu,
                                                              *detail_opencl_buffer(input),
                                                              *detail_opencl_buffer(upstream)),
                                       input.shape(), input.device()));
            return;
        }
        const auto& input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            gradient[index] = input_data[index] > 0.0F ? upstream.data()[index] : 0.0F;
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::relu, *detail_opencl_buffer(input)),
            input.shape(), {input}, std::move(backward), "relu", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(),
                   [](float value) { return std::max(0.0F, value); });
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "relu");
}

Tensor leaky_relu(const Tensor& input, const float negative_slope) {
    if (negative_slope < 0.0F) {
        throw std::invalid_argument("leaky ReLU negative slope must be non-negative");
    }
    return relu(input) - negative_slope * relu(-input);
}

Tensor sigmoid(const Tensor& input) {
    const auto sigmoid_value = [](float value) {
        if (value >= 0.0F) {
            return 1.0F / (1.0F + std::exp(-value));
        }
        const auto exponential = std::exp(value);
        return exponential / (1.0F + exponential);
    };
    auto backward = [input, sigmoid_value](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input, opencl_gradient(opencl::unary_gradient(opencl::UnaryOperation::sigmoid,
                                                              *detail_opencl_buffer(input),
                                                              *detail_opencl_buffer(upstream)),
                                       input.shape(), input.device()));
            return;
        }
        const auto& input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            const auto value = sigmoid_value(input_data[index]);
            gradient[index] = upstream.data()[index] * value * (1.0F - value);
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::sigmoid, *detail_opencl_buffer(input)),
            input.shape(), {input}, std::move(backward), "sigmoid", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(), sigmoid_value);
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "sigmoid");
}

Tensor tanh(const Tensor& input) {
    auto backward = [input](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(input, opencl_gradient(opencl::unary_gradient(
                                                         opencl::UnaryOperation::hyperbolic_tangent,
                                                         *detail_opencl_buffer(input),
                                                         *detail_opencl_buffer(upstream)),
                                                     input.shape(), input.device()));
            return;
        }
        const auto& input_data = input.data();
        std::vector<float> gradient(input.size());
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            const auto value = std::tanh(input_data[index]);
            gradient[index] = upstream.data()[index] * (1.0F - value * value);
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::unary(opencl::UnaryOperation::hyperbolic_tangent, *detail_opencl_buffer(input)),
            input.shape(), {input}, std::move(backward), "tanh", input.device());
    }
    std::vector<float> output(input.size());
    std::transform(input.data().begin(), input.data().end(), output.begin(),
                   [](float value) { return std::tanh(value); });
    return detail_make_result(std::move(output), input.shape(), {input}, std::move(backward),
                              "tanh");
}

Tensor softmax(const Tensor& input, const std::size_t dimension) {
    if (dimension >= input.ndim()) {
        throw std::invalid_argument("softmax dimension is out of range");
    }
    const auto axis_size = input.shape()[dimension];
    std::size_t inner = 1;
    for (std::size_t index = dimension + 1; index < input.ndim(); ++index) {
        inner *= input.shape()[index];
    }
    const auto outer = input.size() / (axis_size * inner);
    const auto compute_values = [input, outer, axis_size, inner]() {
        const auto& input_data = input.data();
        std::vector<float> values(input.size());
        for (std::size_t group = 0; group < outer; ++group) {
            for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t axis = 0; axis < axis_size; ++axis) {
                    const auto index = (group * axis_size + axis) * inner + inner_index;
                    maximum = std::max(maximum, input_data[index]);
                }
                float denominator = 0.0F;
                for (std::size_t axis = 0; axis < axis_size; ++axis) {
                    const auto index = (group * axis_size + axis) * inner + inner_index;
                    values[index] = std::exp(input_data[index] - maximum);
                    denominator += values[index];
                }
                for (std::size_t axis = 0; axis < axis_size; ++axis) {
                    values[(group * axis_size + axis) * inner + inner_index] /= denominator;
                }
            }
        }
        return values;
    };
    std::shared_ptr<opencl::Buffer> gpu_output;
    if (input.device().type() == DeviceType::opencl) {
        gpu_output = opencl::softmax(*detail_opencl_buffer(input), outer, axis_size, inner);
    }
    auto backward = [input, compute_values, outer, axis_size, inner,
                     gpu_output](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (gpu_output != nullptr) {
            detail_accumulate(
                input, opencl_gradient(opencl::softmax_gradient(*gpu_output,
                                                                *detail_opencl_buffer(upstream),
                                                                outer, axis_size, inner),
                                       input.shape(), input.device()));
            return;
        }
        const auto saved_output = compute_values();
        std::vector<float> gradient(input.size());
        for (std::size_t group = 0; group < outer; ++group) {
            for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
                float dot = 0.0F;
                for (std::size_t axis = 0; axis < axis_size; ++axis) {
                    const auto index = (group * axis_size + axis) * inner + inner_index;
                    dot += upstream.data()[index] * saved_output[index];
                }
                for (std::size_t axis = 0; axis < axis_size; ++axis) {
                    const auto index = (group * axis_size + axis) * inner + inner_index;
                    gradient[index] = saved_output[index] * (upstream.data()[index] - dot);
                }
            }
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(std::move(gpu_output), input.shape(), {input},
                                         std::move(backward), "softmax", input.device());
    }
    return detail_make_result(compute_values(), input.shape(), {input}, std::move(backward),
                              "softmax");
}

Tensor reshape(const Tensor& input, Shape shape) {
    if (numel(shape) != input.size()) {
        throw std::invalid_argument("reshape cannot change the number of elements");
    }
    auto backward = [input](const Tensor& upstream) {
        if (input.requires_grad()) {
            if (input.device().type() == DeviceType::opencl) {
                detail_accumulate(input, opencl_gradient(detail_opencl_buffer(upstream),
                                                         input.shape(), input.device()));
            } else {
                detail_accumulate(input, upstream.data());
            }
        }
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(detail_opencl_buffer(input), std::move(shape), {input},
                                         std::move(backward), "reshape", input.device());
    }
    return detail_make_result(input.data(), std::move(shape), {input}, std::move(backward),
                              "reshape");
}

Tensor transpose(const Tensor& input) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("transpose currently requires a rank-2 tensor");
    }
    const auto rows = input.shape()[0];
    const auto columns = input.shape()[1];
    auto backward = [input, rows, columns](const Tensor& upstream) {
        if (!input.requires_grad()) {
            return;
        }
        if (input.device().type() == DeviceType::opencl) {
            detail_accumulate(
                input,
                opencl_gradient(opencl::transpose(*detail_opencl_buffer(upstream), columns, rows),
                                input.shape(), input.device()));
            return;
        }
        std::vector<float> gradient(input.size());
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                gradient[row * columns + column] = upstream.data()[column * rows + row];
            }
        }
        detail_accumulate(input, gradient);
    };
    if (input.device().type() == DeviceType::opencl) {
        return detail_make_opencl_result(
            opencl::transpose(*detail_opencl_buffer(input), rows, columns), {columns, rows},
            {input}, std::move(backward), "transpose", input.device());
    }
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[column * rows + row] = input.data()[row * columns + column];
        }
    }
    return detail_make_result(std::move(output), {columns, rows}, {input}, std::move(backward),
                              "transpose");
}

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    require_same_shape(prediction, target, "mse_loss");
    return mean(pow(prediction - target, 2.0F));
}

Tensor binary_cross_entropy(const Tensor& prediction, const Tensor& target, const float epsilon) {
    require_same_shape(prediction, target, "binary_cross_entropy");
    if (prediction.device() != target.device()) {
        throw std::invalid_argument("binary_cross_entropy tensors must be on the same device");
    }
    if (!(epsilon > 0.0F && epsilon < 0.5F)) {
        throw std::invalid_argument("binary cross entropy epsilon must be in (0, 0.5)");
    }
    if (prediction.device().type() == DeviceType::opencl) {
        auto backward = [prediction, target, epsilon](const Tensor& upstream) {
            auto gradients = opencl::binary_cross_entropy_gradients(
                *detail_opencl_buffer(prediction), *detail_opencl_buffer(target),
                *detail_opencl_buffer(upstream), epsilon, false, prediction.requires_grad(),
                target.requires_grad());
            if (prediction.requires_grad())
                detail_accumulate(prediction,
                                  opencl_gradient(std::move(gradients.prediction),
                                                  prediction.shape(), prediction.device()));
            if (target.requires_grad())
                detail_accumulate(target, opencl_gradient(std::move(gradients.target),
                                                          target.shape(), target.device()));
        };
        return detail_make_opencl_result(
            opencl::binary_cross_entropy_loss(*detail_opencl_buffer(prediction),
                                              *detail_opencl_buffer(target), epsilon, false),
            {}, {prediction, target}, std::move(backward), "binary_cross_entropy",
            prediction.device());
    }
    std::vector<float> clipped(prediction.size());
    std::transform(prediction.data().begin(), prediction.data().end(), clipped.begin(),
                   [epsilon](float value) { return std::clamp(value, epsilon, 1.0F - epsilon); });
    float loss = 0.0F;
    for (std::size_t index = 0; index < clipped.size(); ++index) {
        loss -= target.data()[index] * std::log(clipped[index]) +
                (1.0F - target.data()[index]) * std::log(1.0F - clipped[index]);
    }
    loss /= static_cast<float>(clipped.size());
    return detail_make_result(
        {loss}, {}, {prediction, target},
        [prediction, target, clipped](const Tensor& upstream) {
            const auto scale = upstream.data().front() / static_cast<float>(clipped.size());
            if (prediction.requires_grad()) {
                std::vector<float> gradient(prediction.size());
                for (std::size_t index = 0; index < gradient.size(); ++index) {
                    gradient[index] = scale * (clipped[index] - target.data()[index]) /
                                      (clipped[index] * (1.0F - clipped[index]));
                }
                detail_accumulate(prediction, gradient);
            }
            if (target.requires_grad()) {
                std::vector<float> gradient(target.size());
                for (std::size_t index = 0; index < gradient.size(); ++index) {
                    gradient[index] =
                        scale * (std::log(1.0F - clipped[index]) - std::log(clipped[index]));
                }
                detail_accumulate(target, gradient);
            }
        },
        "binary_cross_entropy");
}

Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target) {
    require_same_shape(logits, target, "binary_cross_entropy_with_logits");
    if (logits.device() != target.device()) {
        throw std::invalid_argument(
            "binary_cross_entropy_with_logits tensors must be on the same device");
    }
    if (logits.device().type() == DeviceType::opencl) {
        auto backward = [logits, target](const Tensor& upstream) {
            auto gradients = opencl::binary_cross_entropy_gradients(
                *detail_opencl_buffer(logits), *detail_opencl_buffer(target),
                *detail_opencl_buffer(upstream), 0.0F, true, logits.requires_grad(),
                target.requires_grad());
            if (logits.requires_grad())
                detail_accumulate(logits, opencl_gradient(std::move(gradients.prediction),
                                                          logits.shape(), logits.device()));
            if (target.requires_grad())
                detail_accumulate(target, opencl_gradient(std::move(gradients.target),
                                                          target.shape(), target.device()));
        };
        return detail_make_opencl_result(
            opencl::binary_cross_entropy_loss(*detail_opencl_buffer(logits),
                                              *detail_opencl_buffer(target), 0.0F, true),
            {}, {logits, target}, std::move(backward), "binary_cross_entropy_with_logits",
            logits.device());
    }
    float loss = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const auto value = logits.data()[index];
        loss += std::max(value, 0.0F) - value * target.data()[index] +
                std::log1p(std::exp(-std::abs(value)));
    }
    loss /= static_cast<float>(logits.size());
    return detail_make_result(
        {loss}, {}, {logits, target},
        [logits, target](const Tensor& upstream) {
            const auto scale = upstream.data().front() / static_cast<float>(logits.size());
            if (logits.requires_grad()) {
                std::vector<float> gradient(logits.size());
                for (std::size_t index = 0; index < gradient.size(); ++index) {
                    const auto value = logits.data()[index];
                    const auto probability = value >= 0.0F
                                                 ? 1.0F / (1.0F + std::exp(-value))
                                                 : std::exp(value) / (1.0F + std::exp(value));
                    gradient[index] = scale * (probability - target.data()[index]);
                }
                detail_accumulate(logits, gradient);
            }
            if (target.requires_grad()) {
                std::vector<float> gradient(target.size());
                for (std::size_t index = 0; index < gradient.size(); ++index) {
                    gradient[index] = -scale * logits.data()[index];
                }
                detail_accumulate(target, gradient);
            }
        },
        "binary_cross_entropy_with_logits");
}

Tensor cross_entropy(const Tensor& logits, const std::vector<std::size_t>& labels) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("cross_entropy logits must have shape [batch, classes]");
    }
    const auto batch = logits.shape()[0];
    const auto classes = logits.shape()[1];
    if (labels.size() != batch) {
        throw std::invalid_argument("cross_entropy requires one label per batch item");
    }
    for (const auto label : labels) {
        if (label >= classes)
            throw std::out_of_range("cross_entropy label is outside the class range");
    }
    if (logits.device().type() == DeviceType::opencl) {
        auto backward = [logits, labels, batch, classes](const Tensor& upstream) {
            if (logits.requires_grad()) {
                detail_accumulate(
                    logits, opencl_gradient(opencl::cross_entropy_gradient(
                                                *detail_opencl_buffer(logits), labels,
                                                *detail_opencl_buffer(upstream), batch, classes),
                                            logits.shape(), logits.device()));
            }
        };
        return detail_make_opencl_result(
            opencl::cross_entropy_loss(*detail_opencl_buffer(logits), labels, batch, classes), {},
            {logits}, std::move(backward), "cross_entropy", logits.device());
    }
    std::vector<float> probabilities(logits.size());
    float loss = 0.0F;
    for (std::size_t row = 0; row < batch; ++row) {
        if (labels[row] >= classes) {
            throw std::out_of_range("cross_entropy label is outside the class range");
        }
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t column = 0; column < classes; ++column) {
            maximum = std::max(maximum, logits.data()[row * classes + column]);
        }
        float denominator = 0.0F;
        for (std::size_t column = 0; column < classes; ++column) {
            probabilities[row * classes + column] =
                std::exp(logits.data()[row * classes + column] - maximum);
            denominator += probabilities[row * classes + column];
        }
        for (std::size_t column = 0; column < classes; ++column) {
            probabilities[row * classes + column] /= denominator;
        }
        loss -= std::log(probabilities[row * classes + labels[row]]);
    }
    loss /= static_cast<float>(batch);
    return detail_make_result(
        {loss}, {}, {logits},
        [logits, labels, probabilities, batch, classes](const Tensor& upstream) {
            if (!logits.requires_grad()) {
                return;
            }
            auto gradient = probabilities;
            for (std::size_t row = 0; row < batch; ++row) {
                gradient[row * classes + labels[row]] -= 1.0F;
            }
            const auto scale = upstream.data().front() / static_cast<float>(batch);
            for (auto& value : gradient) {
                value *= scale;
            }
            detail_accumulate(logits, gradient);
        },
        "cross_entropy");
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}
Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return subtract(lhs, rhs);
}
Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return multiply(lhs, rhs);
}
Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return divide(lhs, rhs);
}
Tensor operator-(const Tensor& input) {
    return negate(input);
}
Tensor operator+(const Tensor& lhs, float rhs) {
    return lhs + Tensor::scalar(rhs, false, {}, lhs.device());
}
Tensor operator+(float lhs, const Tensor& rhs) {
    return Tensor::scalar(lhs, false, {}, rhs.device()) + rhs;
}
Tensor operator-(const Tensor& lhs, float rhs) {
    return lhs - Tensor::scalar(rhs, false, {}, lhs.device());
}
Tensor operator-(float lhs, const Tensor& rhs) {
    return Tensor::scalar(lhs, false, {}, rhs.device()) - rhs;
}
Tensor operator*(const Tensor& lhs, float rhs) {
    return lhs * Tensor::scalar(rhs, false, {}, lhs.device());
}
Tensor operator*(float lhs, const Tensor& rhs) {
    return Tensor::scalar(lhs, false, {}, rhs.device()) * rhs;
}
Tensor operator/(const Tensor& lhs, float rhs) {
    return lhs / Tensor::scalar(rhs, false, {}, lhs.device());
}
Tensor operator/(float lhs, const Tensor& rhs) {
    return Tensor::scalar(lhs, false, {}, rhs.device()) / rhs;
}

} // namespace clnn
