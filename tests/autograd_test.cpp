#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 const float tolerance = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
    }
}

TEST(Tensor, ValidatesShapeAndScalarAccess) {
    EXPECT_THROW((clnn::Tensor({1.0F}, {2})), std::invalid_argument);
    EXPECT_THROW((clnn::Tensor::zeros({0, 2})), std::invalid_argument);
    EXPECT_THROW((clnn::Tensor({1.0F, 2.0F}, {2}).item()), std::logic_error);
    EXPECT_FLOAT_EQ(clnn::Tensor::scalar(3.0F).item(), 3.0F);
}

TEST(Autograd, DifferentiatesAndAccumulatesAcrossBranches) {
    clnn::Tensor x({-2.0F, 0.5F, 3.0F}, {3}, true);
    auto output = clnn::sum(x * x + 2.0F * x);
    output.backward();
    expect_near(x.grad(), {-2.0F, 3.0F, 8.0F});
}

TEST(Autograd, ReducesBroadcastGradients) {
    clnn::Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    clnn::Tensor bias({0.5F, -1.0F, 2.0F}, {3}, true);
    auto output = clnn::mean(matrix + bias);
    output.backward();
    expect_near(matrix.grad(), std::vector<float>(6, 1.0F / 6.0F));
    expect_near(bias.grad(), std::vector<float>(3, 1.0F / 3.0F));
}

TEST(Autograd, ComputesMatrixMultiplicationGradients) {
    clnn::Tensor lhs({1, 2, 3, 4}, {2, 2}, true);
    clnn::Tensor rhs({5, 6, 7, 8}, {2, 2}, true);
    clnn::sum(clnn::matmul(lhs, rhs)).backward();
    expect_near(lhs.grad(), {11, 15, 11, 15});
    expect_near(rhs.grad(), {4, 4, 6, 6});
}

TEST(Autograd, ComputesConvolutionForwardAndBackward) {
    clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3}, true);
    clnn::Tensor weight({1, 0, 0, -1}, {1, 1, 2, 2}, true);
    clnn::Tensor bias({0.5F}, {1}, true);
    auto output = clnn::conv2d(input, weight, bias);
    expect_near(output.data(), {-3.5F, -3.5F, -3.5F, -3.5F});
    clnn::sum(output).backward();
    expect_near(input.grad(), {1, 1, 0, 1, 0, -1, 0, -1, -1});
    expect_near(weight.grad(), {12, 16, 24, 28});
    expect_near(bias.grad(), {4});
}

TEST(Autograd, SupportsExplicitVectorJacobianProducts) {
    clnn::Tensor input({1, 2}, {2}, true);
    auto output = input * input;
    output.backward(clnn::Tensor({3, 4}, {2}));
    expect_near(input.grad(), {6, 16});
}

TEST(Autograd, RejectsUnsafeGraphReuseAndMutation) {
    clnn::Tensor input({2.0F}, {1}, true);
    auto output = clnn::sum(input * input);
    input.set_data({3.0F});
    EXPECT_THROW(output.backward(), std::logic_error);

    clnn::Tensor second({2.0F}, {1}, true);
    auto second_output = clnn::sum(second * second);
    second_output.backward();
    EXPECT_THROW(second_output.backward(), std::logic_error);
}

TEST(Autograd, NoGradGuardDisablesGraphConstruction) {
    clnn::Tensor input({1.0F}, {1}, true);
    {
        clnn::NoGradGuard guard;
        EXPECT_FALSE((input * input).requires_grad());
    }
    EXPECT_TRUE((input * input).requires_grad());
}

TEST(Autograd, NonlinearGradientMatchesFiniteDifference) {
    const std::vector<float> values{-1.2F, -0.1F, 0.7F, 2.0F};
    clnn::Tensor input(values, {4}, true);
    clnn::sum(clnn::tanh(input) * clnn::sigmoid(input)).backward();

    constexpr float epsilon = 1.0e-3F;
    std::vector<float> numerical(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        auto lower = values;
        auto upper = values;
        lower[index] -= epsilon;
        upper[index] += epsilon;
        const auto lower_value = clnn::sum(clnn::tanh(clnn::Tensor(lower, {4})) *
                                           clnn::sigmoid(clnn::Tensor(lower, {4})))
                                     .item();
        const auto upper_value = clnn::sum(clnn::tanh(clnn::Tensor(upper, {4})) *
                                           clnn::sigmoid(clnn::Tensor(upper, {4})))
                                     .item();
        numerical[index] = (upper_value - lower_value) / (2.0F * epsilon);
    }
    expect_near(input.grad(), numerical, 2.0e-4F);
}

} // namespace
