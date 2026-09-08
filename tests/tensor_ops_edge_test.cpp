#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 const float tolerance = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
    }
}

TEST(TensorEdge, FactoriesMetadataAndUndefinedState) {
    EXPECT_NO_THROW(clnn::Device::cpu().synchronize());
    const clnn::Tensor undefined;
    EXPECT_FALSE(undefined.defined());
    EXPECT_THROW(static_cast<void>(undefined.shape()), std::logic_error);

    const auto zeros = clnn::Tensor::zeros({2, 2}, false, "zeros");
    const auto ones = clnn::Tensor::ones({2, 2});
    EXPECT_EQ(zeros.data(), (std::vector<float>{0, 0, 0, 0}));
    EXPECT_EQ(ones.data(), (std::vector<float>{1, 1, 1, 1}));
    EXPECT_EQ(zeros.name(), "zeros");
    EXPECT_EQ(zeros.ndim(), 2U);
    EXPECT_EQ(zeros.size(), 4U);
    EXPECT_FALSE(zeros.is_scalar());
    EXPECT_TRUE(clnn::Tensor::scalar(1.0F).is_scalar());
    EXPECT_EQ(clnn::shape_string({2, 3, 4}), "[2, 3, 4]");
    EXPECT_EQ(clnn::numel({}), 1U);
    EXPECT_THROW(clnn::numel({std::numeric_limits<std::size_t>::max(), 2}), std::overflow_error);

    std::mt19937 first_generator(123);
    std::mt19937 second_generator(123);
    EXPECT_EQ(clnn::Tensor::randn({8}, first_generator).data(),
              clnn::Tensor::randn({8}, second_generator).data());
    EXPECT_THROW(clnn::Tensor::randn({1}, first_generator, 0.0F, 0.0F), std::invalid_argument);
}

TEST(TensorEdge, MutationDetachTransferAndLeafBackward) {
    clnn::Tensor tensor({1, 2}, {2}, true, "source");
    EXPECT_THROW(tensor.set_data({1}), std::invalid_argument);
    auto detached = tensor.detach();
    tensor.set_data({3, 4});
    EXPECT_EQ(detached.data(), (std::vector<float>{1, 2}));
    EXPECT_FALSE(detached.requires_grad());
    const auto copied = tensor.to(clnn::Device::cpu());
    EXPECT_EQ(copied.data(), tensor.data());
    EXPECT_EQ(copied.name(), "source");

    auto scalar = clnn::Tensor::scalar(2.0F, true);
    scalar.backward();
    scalar.backward();
    EXPECT_EQ(scalar.grad(), (std::vector<float>{2}));
    scalar.zero_grad();
    EXPECT_FALSE(scalar.has_grad());

    EXPECT_THROW(detached.backward(), std::logic_error);
    EXPECT_THROW(tensor.backward(), std::logic_error);
    EXPECT_THROW(tensor.backward(clnn::Tensor::scalar(1.0F)), std::invalid_argument);
}

TEST(TensorEdge, RetainsGraphsAndProtectsOperationResults) {
    clnn::Tensor input({2, 3}, {2}, true);
    auto result = clnn::sum(input * input);
    EXPECT_THROW(result.set_data({2}), std::logic_error);
    result.backward(std::nullopt, true);
    expect_near(input.grad(), {4, 6});
    result.backward();
    expect_near(input.grad(), {8, 12});
    EXPECT_THROW(result.set_data({2}), std::logic_error);

    clnn::detail_accumulate(input, {1, 1});
    expect_near(input.grad(), {9, 13});
    EXPECT_THROW(clnn::detail_accumulate(input, {1}), std::logic_error);
    clnn::detail_accumulate(clnn::Tensor::ones({2}), {1, 1});
    EXPECT_THROW(static_cast<void>(clnn::detail_opencl_buffer(input)), std::logic_error);
}

TEST(OperationsEdge, ArithmeticAndReflectedScalarOperatorsDifferentiate) {
    clnn::Tensor lhs({2, 4}, {2}, true);
    clnn::Tensor rhs({1, 2}, {2}, true);
    clnn::sum((lhs - rhs) + lhs / rhs).backward();
    expect_near(lhs.grad(), {2.0F, 1.5F});
    expect_near(rhs.grad(), {-3.0F, -2.0F});

    const clnn::Tensor values({2, 4}, {2});
    EXPECT_EQ((3.0F + values).data(), (std::vector<float>{5, 7}));
    EXPECT_EQ((5.0F - values).data(), (std::vector<float>{3, 1}));
    EXPECT_EQ((3.0F * values).data(), (std::vector<float>{6, 12}));
    expect_near((8.0F / values).data(), {4, 2});
    EXPECT_EQ((-values).data(), (std::vector<float>{-2, -4}));
}

TEST(OperationsEdge, RejectsInvalidBroadcastAndMatrixShapes) {
    const clnn::Tensor matrix({1, 2, 3, 4}, {2, 2});
    const clnn::Tensor vector({1, 2, 3}, {3});
    EXPECT_THROW(static_cast<void>(matrix + vector), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::matmul(vector, vector)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::matmul(matrix, clnn::Tensor::ones({3, 2}))),
                 std::invalid_argument);
}

TEST(OperationsEdge, ValidatesConvolutionArguments) {
    const clnn::Tensor input = clnn::Tensor::ones({1, 1, 3, 3});
    const clnn::Tensor weight = clnn::Tensor::ones({1, 1, 2, 2});
    EXPECT_THROW(static_cast<void>(clnn::conv2d(clnn::Tensor::ones({3, 3}), weight)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::conv2d(input, clnn::Tensor::ones({1, 2, 2, 2}))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::conv2d(input, weight, std::nullopt, {0, 1})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::conv2d(input, clnn::Tensor::ones({1, 1, 4, 4}))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::conv2d(input, weight, clnn::Tensor::zeros({2}))),
                 std::invalid_argument);
}

TEST(OperationsEdge, ConvolutionSupportsPaddingStrideAndSelectiveGradients) {
    clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3}, true);
    const clnn::Tensor weight({1, 0, 0, -1}, {1, 1, 2, 2});
    const auto output = clnn::conv2d(input, weight, std::nullopt, {2, 2}, {1, 1});
    EXPECT_EQ(output.shape(), (clnn::Shape{1, 1, 2, 2}));
    clnn::sum(output).backward();
    EXPECT_TRUE(input.has_grad());
    EXPECT_FALSE(weight.has_grad());

    const clnn::Tensor fixed_input({1, 2, 3, 4}, {1, 1, 2, 2});
    clnn::Tensor trainable_weight({2}, {1, 1, 1, 1}, true);
    clnn::sum(clnn::conv2d(fixed_input, trainable_weight)).backward();
    EXPECT_EQ(trainable_weight.grad(), (std::vector<float>{10}));
}

TEST(OperationsEdge, UnaryFunctionsHaveExpectedValuesAndGradients) {
    clnn::Tensor positive({0.5F, 2.0F}, {2}, true);
    clnn::sum(clnn::exp(clnn::log(positive)) + clnn::pow(positive, 3.0F)).backward();
    expect_near(positive.grad(), {1.75F, 13.0F}, 2.0e-5F);
    EXPECT_THROW(static_cast<void>(clnn::log(clnn::Tensor({0, 1}, {2}))), std::domain_error);
    EXPECT_THROW(static_cast<void>(clnn::leaky_relu(positive, -0.1F)), std::invalid_argument);

    clnn::Tensor signed_values({-2, 0, 3}, {3}, true);
    clnn::sum(clnn::relu(signed_values) + clnn::leaky_relu(signed_values, 0.2F)).backward();
    expect_near(signed_values.grad(), {0.2F, 0.0F, 2.0F});
}

TEST(OperationsEdge, SoftmaxReshapeAndTransposeCoverGeneralAxes) {
    clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}, true);
    auto probabilities = clnn::softmax(input, 1);
    for (std::size_t outer = 0; outer < 2; ++outer) {
        for (std::size_t inner = 0; inner < 2; ++inner) {
            const auto first = probabilities.data()[(outer * 2) * 2 + inner];
            const auto second = probabilities.data()[(outer * 2 + 1) * 2 + inner];
            EXPECT_NEAR(first + second, 1.0F, 1.0e-6F);
        }
    }
    probabilities.backward(clnn::Tensor::ones({2, 2, 2}));
    expect_near(input.grad(), std::vector<float>(8, 0.0F), 1.0e-5F);
    EXPECT_THROW(static_cast<void>(clnn::softmax(input, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::reshape(input, {3, 3})), std::invalid_argument);

    clnn::Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    auto transposed = clnn::transpose(matrix);
    EXPECT_EQ(transposed.shape(), (clnn::Shape{3, 2}));
    EXPECT_EQ(transposed.data(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    clnn::sum(transposed).backward();
    EXPECT_EQ(matrix.grad(), (std::vector<float>{1, 1, 1, 1, 1, 1}));
    EXPECT_THROW(static_cast<void>(clnn::transpose(input)), std::invalid_argument);
}

TEST(LossesEdge, BinaryCrossEntropyDifferentiatesBothInputs) {
    clnn::Tensor prediction({0.2F, 0.8F}, {2}, true);
    clnn::Tensor target({0.0F, 1.0F}, {2}, true);
    auto loss = clnn::binary_cross_entropy(prediction, target);
    EXPECT_NEAR(loss.item(), -std::log(0.8F), 1.0e-6F);
    loss.backward();
    expect_near(prediction.grad(), {0.625F, -0.625F});
    EXPECT_TRUE(target.has_grad());

    EXPECT_THROW(
        static_cast<void>(clnn::binary_cross_entropy(prediction, clnn::Tensor::zeros({1}))),
        std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::binary_cross_entropy(prediction, target, 0.0F)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::binary_cross_entropy(prediction, target, 0.5F)),
                 std::invalid_argument);
}

TEST(LossesEdge, LogitTargetsDifferentiateAndCrossEntropyValidatesRank) {
    const clnn::Tensor logits_values({-2, 3}, {2});
    clnn::Tensor logits(logits_values.data(), logits_values.shape(), true);
    clnn::Tensor targets({0, 1}, {2}, true);
    clnn::binary_cross_entropy_with_logits(logits, targets).backward();
    EXPECT_TRUE(logits.has_grad());
    expect_near(targets.grad(), {1.0F, -1.5F});
    EXPECT_THROW(static_cast<void>(clnn::cross_entropy(logits_values, {0, 1})),
                 std::invalid_argument);
}

TEST(OperationsEdge, RejectsCrossDeviceOperandsWhenOpenClExists) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    const clnn::Tensor cpu = clnn::Tensor::ones({2, 2});
    const auto on_gpu = cpu.to(gpu);
    EXPECT_THROW(static_cast<void>(cpu + on_gpu), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::matmul(cpu, on_gpu)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::conv2d(clnn::Tensor::ones({1, 1, 2, 2}),
                                                clnn::Tensor::ones({1, 1, 1, 1}).to(gpu))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::binary_cross_entropy(cpu, on_gpu)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::binary_cross_entropy_with_logits(cpu, on_gpu)),
                 std::invalid_argument);
}

} // namespace
