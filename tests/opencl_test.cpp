#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>

namespace {

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 const float tolerance = 1.0e-4F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
    }
}

void expect_parameter_gradients_near(clnn::nn::Module& actual, clnn::nn::Module& expected,
                                     const float tolerance = 1.0e-4F) {
    const auto actual_parameters = actual.named_parameters();
    const auto expected_parameters = expected.named_parameters();
    ASSERT_EQ(actual_parameters.size(), expected_parameters.size());
    for (std::size_t index = 0; index < actual_parameters.size(); ++index) {
        EXPECT_EQ(actual_parameters[index].name, expected_parameters[index].name);
        expect_near(actual_parameters[index].tensor->grad(),
                    expected_parameters[index].tensor->grad(), tolerance);
    }
}

TEST(OpenCL, ReportsAndExecutesOnGpu) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    EXPECT_FALSE(gpu.name().empty());
    clnn::Tensor lhs({1, 2, 3, 4, 5, 6}, {2, 3}, true, {}, gpu);
    clnn::Tensor rhs({2, 0, -1}, {3}, true, {}, gpu);
    auto result = clnn::sum(clnn::tanh(lhs * rhs + 1.0F));
    result.backward();
    EXPECT_NO_THROW(gpu.synchronize());
    EXPECT_EQ(result.device(), gpu);
    EXPECT_TRUE(std::isfinite(result.item()));
    EXPECT_EQ(lhs.grad().size(), lhs.size());
    EXPECT_EQ(rhs.grad().size(), rhs.size());
}

TEST(OpenCL, MatchesCpuForMatrixAndConvolution) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    const clnn::Tensor cpu_lhs({1, 2, 3, 4}, {2, 2});
    const clnn::Tensor cpu_rhs({5, 6, 7, 8}, {2, 2});
    expect_near(clnn::matmul(cpu_lhs.to(gpu), cpu_rhs.to(gpu)).data(),
                clnn::matmul(cpu_lhs, cpu_rhs).data());

    const clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
    const clnn::Tensor weight({1, 0, 0, -1}, {1, 1, 2, 2});
    const clnn::Tensor bias({0.5F}, {1});
    expect_near(clnn::conv2d(input.to(gpu), weight.to(gpu), bias.to(gpu)).data(),
                clnn::conv2d(input, weight, bias).data());
}

TEST(OpenCL, BackwardKernelsMatchCpuReference) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    clnn::Tensor cpu_lhs({1, -2, 3, 0.5F, 4, -1}, {2, 3}, true);
    clnn::Tensor cpu_rhs({2, -1, 0.25F}, {3}, true);
    auto gpu_lhs = cpu_lhs.detach().to(gpu);
    gpu_lhs = clnn::Tensor(gpu_lhs.data(), gpu_lhs.shape(), true, {}, gpu);
    auto gpu_rhs = clnn::Tensor(cpu_rhs.data(), cpu_rhs.shape(), true, {}, gpu);
    clnn::mean(clnn::tanh(cpu_lhs * cpu_rhs + 0.5F)).backward();
    clnn::mean(clnn::tanh(gpu_lhs * gpu_rhs + 0.5F)).backward();
    expect_near(gpu_lhs.grad(), cpu_lhs.grad());
    expect_near(gpu_rhs.grad(), cpu_rhs.grad());

    clnn::Tensor cpu_input({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3}, true);
    clnn::Tensor cpu_weight({1, 0, 0, -1}, {1, 1, 2, 2}, true);
    auto gpu_input = clnn::Tensor(cpu_input.data(), cpu_input.shape(), true, {}, gpu);
    auto gpu_weight = clnn::Tensor(cpu_weight.data(), cpu_weight.shape(), true, {}, gpu);
    clnn::sum(clnn::conv2d(cpu_input, cpu_weight)).backward();
    clnn::sum(clnn::conv2d(gpu_input, gpu_weight)).backward();
    expect_near(gpu_input.grad(), cpu_input.grad());
    expect_near(gpu_weight.grad(), cpu_weight.grad());
}

TEST(OpenCL, SequentialDenseLayersExecuteInOrderWithoutIntermediateSynchronization) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    std::mt19937 cpu_generator(2026);
    std::mt19937 gpu_generator(2026);

    clnn::nn::Sequential cpu_model;
    cpu_model.add(std::make_unique<clnn::nn::Linear>(4, 8, cpu_generator))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::Linear>(8, 6, cpu_generator))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::Linear>(6, 3, cpu_generator))
        .add(std::make_unique<clnn::nn::Sigmoid>());
    clnn::nn::Sequential gpu_model;
    gpu_model.add(std::make_unique<clnn::nn::Linear>(4, 8, gpu_generator, true, gpu))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::Linear>(8, 6, gpu_generator, true, gpu))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::Linear>(6, 3, gpu_generator, true, gpu))
        .add(std::make_unique<clnn::nn::Sigmoid>());

    const std::vector<float> input_values{
        -1.0F, 0.5F, 2.0F,  -0.5F, 0.25F, -0.75F, 1.25F, 0.5F,  0.0F, 1.0F,
        -1.5F, 2.5F, -0.2F, 0.3F,  0.7F,  -1.1F,  1.5F,  -0.4F, 0.8F, -2.0F,
    };
    const std::vector<float> upstream_values{
        1.0F, -0.5F, 0.25F, -1.0F, 0.75F, 0.5F, 0.2F,  -0.3F,
        0.9F, -0.8F, 0.4F,  1.2F,  -0.6F, 0.1F, -1.1F,
    };
    clnn::Tensor cpu_input(input_values, {5, 4}, true);
    clnn::Tensor gpu_input(input_values, {5, 4}, true, {}, gpu);
    const clnn::Tensor cpu_upstream(upstream_values, {5, 3});
    const clnn::Tensor gpu_upstream(upstream_values, {5, 3}, false, {}, gpu);

    auto cpu_output = cpu_model(cpu_input);
    cpu_output.backward(cpu_upstream);

    auto gpu_output = gpu_model(gpu_input);
    gpu_output.backward(gpu_upstream);
    gpu.synchronize();

    expect_near(gpu_output.data(), cpu_output.data(), 3.0e-4F);
    expect_near(gpu_input.grad(), cpu_input.grad(), 5.0e-4F);
    expect_parameter_gradients_near(gpu_model, cpu_model, 5.0e-4F);
}

TEST(OpenCL, SequentialConvolutionLayersExecuteInOrderWithoutIntermediateSynchronization) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    std::mt19937 cpu_generator(31415);
    std::mt19937 gpu_generator(31415);

    clnn::nn::Sequential cpu_model;
    cpu_model
        .add(std::make_unique<clnn::nn::Conv2d>(1, 2, std::array<std::size_t, 2>{3, 3},
                                                cpu_generator, std::array<std::size_t, 2>{1, 1},
                                                clnn::nn::PaddingMode::same))
        .add(std::make_unique<clnn::nn::BatchNorm>(2))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::MaxPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::Conv2d>(2, 3, std::array<std::size_t, 2>{3, 3},
                                                cpu_generator, std::array<std::size_t, 2>{1, 1},
                                                clnn::nn::PaddingMode::same))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::AvgPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::Flatten>())
        .add(std::make_unique<clnn::nn::Linear>(3, 5, cpu_generator))
        .add(std::make_unique<clnn::nn::Dropout>(0.2F, 99))
        .add(std::make_unique<clnn::nn::Softmax>(1));
    clnn::nn::Sequential gpu_model;
    gpu_model
        .add(std::make_unique<clnn::nn::Conv2d>(1, 2, std::array<std::size_t, 2>{3, 3},
                                                gpu_generator, std::array<std::size_t, 2>{1, 1},
                                                clnn::nn::PaddingMode::same, true, gpu))
        .add(std::make_unique<clnn::nn::BatchNorm>(2, 1.0e-5F, 0.1F, true, gpu))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::MaxPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::Conv2d>(2, 3, std::array<std::size_t, 2>{3, 3},
                                                gpu_generator, std::array<std::size_t, 2>{1, 1},
                                                clnn::nn::PaddingMode::same, true, gpu))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::AvgPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::Flatten>())
        .add(std::make_unique<clnn::nn::Linear>(3, 5, gpu_generator, true, gpu))
        .add(std::make_unique<clnn::nn::Dropout>(0.2F, 99))
        .add(std::make_unique<clnn::nn::Softmax>(1));

    const std::vector<float> input_values{
        -1.0F, -0.8F, -0.6F, -0.4F, -0.2F, 0.0F,  0.2F,  0.4F,  0.6F,  0.8F,  1.0F,
        1.2F,  1.4F,  1.6F,  1.8F,  2.0F,  1.5F,  1.2F,  0.9F,  0.6F,  0.3F,  0.0F,
        -0.3F, -0.6F, -0.9F, -1.2F, -1.5F, -1.8F, -2.1F, -2.4F, -2.7F, -3.0F,
    };
    const std::vector<float> upstream_values{
        1.0F, -0.5F, 0.25F, 0.75F, -1.0F, -0.2F, 0.4F, -0.6F, 0.8F, 0.1F,
    };
    clnn::Tensor cpu_input(input_values, {2, 1, 4, 4}, true);
    clnn::Tensor gpu_input(input_values, {2, 1, 4, 4}, true, {}, gpu);
    const clnn::Tensor cpu_upstream(upstream_values, {2, 5});
    const clnn::Tensor gpu_upstream(upstream_values, {2, 5}, false, {}, gpu);

    auto cpu_output = cpu_model(cpu_input);
    cpu_output.backward(cpu_upstream);

    auto gpu_output = gpu_model(gpu_input);
    gpu_output.backward(gpu_upstream);
    gpu.synchronize();

    expect_near(gpu_output.data(), cpu_output.data(), 5.0e-4F);
    expect_near(gpu_input.grad(), cpu_input.grad(), 1.0e-3F);
    expect_parameter_gradients_near(gpu_model, cpu_model, 1.0e-3F);
}

TEST(OpenCL, LossAndOptimizerKernelsMatchCpuReference) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    clnn::Tensor cpu({-2, 0.5F, 3}, {3}, true);
    clnn::Tensor gpu_tensor(cpu.data(), cpu.shape(), true, {}, gpu);
    const clnn::Tensor cpu_target({0, 1, 1}, {3});
    const clnn::Tensor gpu_target(cpu_target.data(), cpu_target.shape(), false, {}, gpu);
    auto cpu_loss = clnn::binary_cross_entropy_with_logits(cpu, cpu_target);
    auto gpu_loss = clnn::binary_cross_entropy_with_logits(gpu_tensor, gpu_target);
    EXPECT_NEAR(gpu_loss.item(), cpu_loss.item(), 1.0e-5F);
    cpu_loss.backward();
    gpu_loss.backward();
    expect_near(gpu_tensor.grad(), cpu.grad());
    clnn::optim::AdamW cpu_optimizer({&cpu}, 0.01F);
    clnn::optim::AdamW gpu_optimizer({&gpu_tensor}, 0.01F);
    cpu_optimizer.step();
    gpu_optimizer.step();
    expect_near(gpu_tensor.data(), cpu.data());
}

TEST(OpenCL, UnaryTransformAndSoftmaxKernelsMatchCpuReference) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    clnn::Tensor cpu({0.25F, 0.75F, 1.5F, 2.0F}, {2, 2}, true);
    clnn::Tensor on_gpu(cpu.data(), cpu.shape(), true, {}, gpu);
    const auto expression = [](const clnn::Tensor& input) {
        return clnn::sum(clnn::pow(input, 2.5F) + clnn::exp(input) + clnn::log(input) +
                         clnn::relu(input) + clnn::sigmoid(input) + clnn::tanh(input));
    };
    auto cpu_loss = expression(cpu);
    auto gpu_loss = expression(on_gpu);
    EXPECT_NEAR(gpu_loss.item(), cpu_loss.item(), 2.0e-4F);
    cpu_loss.backward();
    gpu_loss.backward();
    expect_near(on_gpu.grad(), cpu.grad(), 3.0e-4F);

    clnn::Tensor cpu_softmax({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}, true);
    clnn::Tensor gpu_softmax(cpu_softmax.data(), cpu_softmax.shape(), true, {}, gpu);
    const clnn::Tensor upstream({1, -1, 2, -2, 3, -3, 4, -4}, {2, 2, 2});
    auto cpu_probabilities = clnn::softmax(cpu_softmax, 1);
    auto gpu_probabilities = clnn::softmax(gpu_softmax, 1);
    expect_near(gpu_probabilities.data(), cpu_probabilities.data());
    cpu_probabilities.backward(upstream);
    gpu_probabilities.backward(upstream.to(gpu));
    expect_near(gpu_softmax.grad(), cpu_softmax.grad());

    clnn::Tensor cpu_matrix({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    clnn::Tensor gpu_matrix(cpu_matrix.data(), cpu_matrix.shape(), true, {}, gpu);
    auto cpu_transposed = clnn::reshape(clnn::transpose(cpu_matrix), {6});
    auto gpu_transposed = clnn::reshape(clnn::transpose(gpu_matrix), {6});
    expect_near(gpu_transposed.data(), cpu_transposed.data());
    clnn::sum(cpu_transposed).backward();
    clnn::sum(gpu_transposed).backward();
    expect_near(gpu_matrix.grad(), cpu_matrix.grad());
}

TEST(OpenCL, RemainingLossAndOptimizerKernelsMatchCpuReference) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();

    clnn::Tensor cpu_prediction({0.2F, 0.7F, 0.9F}, {3}, true);
    clnn::Tensor cpu_target({0, 1, 1}, {3}, true);
    clnn::Tensor gpu_prediction(cpu_prediction.data(), {3}, true, {}, gpu);
    clnn::Tensor gpu_target(cpu_target.data(), {3}, true, {}, gpu);
    auto cpu_bce = clnn::binary_cross_entropy(cpu_prediction, cpu_target);
    auto gpu_bce = clnn::binary_cross_entropy(gpu_prediction, gpu_target);
    EXPECT_NEAR(gpu_bce.item(), cpu_bce.item(), 1.0e-5F);
    cpu_bce.backward();
    gpu_bce.backward();
    expect_near(gpu_prediction.grad(), cpu_prediction.grad());
    expect_near(gpu_target.grad(), cpu_target.grad());

    clnn::Tensor cpu_logits({2, 1, 0, -1, 0, 1}, {2, 3}, true);
    clnn::Tensor gpu_logits(cpu_logits.data(), cpu_logits.shape(), true, {}, gpu);
    auto cpu_cross_entropy = clnn::cross_entropy(cpu_logits, {0, 2});
    auto gpu_cross_entropy = clnn::cross_entropy(gpu_logits, {0, 2});
    EXPECT_NEAR(gpu_cross_entropy.item(), cpu_cross_entropy.item(), 1.0e-5F);
    cpu_cross_entropy.backward();
    gpu_cross_entropy.backward();
    expect_near(gpu_logits.grad(), cpu_logits.grad());

    clnn::Tensor cpu_sgd({1, -2}, {2}, true);
    clnn::Tensor gpu_sgd(cpu_sgd.data(), {2}, true, {}, gpu);
    clnn::optim::SGD cpu_sgd_optimizer({&cpu_sgd}, 0.05F, 0.5F, 0.1F);
    clnn::optim::SGD gpu_sgd_optimizer({&gpu_sgd}, 0.05F, 0.5F, 0.1F);
    for (int step = 0; step < 2; ++step) {
        cpu_sgd_optimizer.zero_grad();
        gpu_sgd_optimizer.zero_grad();
        clnn::sum(cpu_sgd * cpu_sgd).backward();
        clnn::sum(gpu_sgd * gpu_sgd).backward();
        cpu_sgd_optimizer.step();
        gpu_sgd_optimizer.step();
    }
    expect_near(gpu_sgd.data(), cpu_sgd.data());

    clnn::Tensor cpu_adam({1, -2}, {2}, true);
    clnn::Tensor gpu_adam(cpu_adam.data(), {2}, true, {}, gpu);
    clnn::optim::Adam cpu_adam_optimizer({&cpu_adam}, 0.01F, 0.8F, 0.9F, 1.0e-6F, 0.1F);
    clnn::optim::Adam gpu_adam_optimizer({&gpu_adam}, 0.01F, 0.8F, 0.9F, 1.0e-6F, 0.1F);
    clnn::sum(cpu_adam * cpu_adam).backward();
    clnn::sum(gpu_adam * gpu_adam).backward();
    cpu_adam_optimizer.step();
    gpu_adam_optimizer.step();
    expect_near(gpu_adam.data(), cpu_adam.data());
}

TEST(OpenCL, OptimizerStateRoundTripsFromResidentBuffers) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    clnn::Tensor original({1.0F, -2.0F}, {2}, true, {}, gpu);
    clnn::optim::Adam optimizer({&original}, 0.01F);
    clnn::sum(original * original).backward();
    optimizer.step();
    optimizer.zero_grad();

    const auto path = std::filesystem::temp_directory_path() / "clnn_gpu_optimizer_state.bin";
    clnn::optim::save_state_dict(optimizer, path);
    clnn::Tensor restored_parameter(original.data(), original.shape(), true, {}, gpu);
    clnn::optim::Adam restored({&restored_parameter}, 0.5F);
    clnn::optim::load_state_dict(restored, path);
    std::filesystem::remove(path);

    clnn::sum(original * original).backward();
    optimizer.step();
    clnn::sum(restored_parameter * restored_parameter).backward();
    restored.step();
    expect_near(restored_parameter.data(), original.data());
}

TEST(OpenCL, DeviceValidationAndGpuMutationAreExplicit) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    EXPECT_THROW(
        static_cast<void>(clnn::Device::opencl(std::numeric_limits<std::size_t>::max(), 0).name()),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(clnn::Device::opencl(0, std::numeric_limits<std::size_t>::max()).name()),
        std::out_of_range);

    clnn::Tensor tensor({1, 2}, {2}, true, "gpu", clnn::Device::opencl());
    tensor.set_data({3, 4});
    EXPECT_EQ(tensor.data(), (std::vector<float>{3, 4}));
    EXPECT_EQ(tensor.name(), "gpu");
}

} // namespace
