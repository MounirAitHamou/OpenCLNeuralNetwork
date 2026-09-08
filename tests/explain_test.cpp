#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace {

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 const float tolerance = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
}

clnn::nn::Linear linear_model(const clnn::Device device = clnn::Device::cpu()) {
    std::mt19937 generator(1);
    clnn::nn::Linear model(2, 2, generator, true, device);
    model.weight().set_data({2.0F, -1.0F, -3.0F, 4.0F});
    model.bias().set_data({0.5F, -0.25F});
    return model;
}

TEST(Explain, SaliencyAndInputTimesGradientAreExactForLinearModel) {
    auto model = linear_model();
    const clnn::Tensor input({1.5F, -2.0F}, {1, 2});

    const auto saliency = clnn::explain::saliency(model, input, 0);
    const auto input_times_gradient = clnn::explain::input_x_gradient(model, input, 0);

    EXPECT_EQ(saliency.shape(), input.shape());
    expect_near(saliency.data(), {2.0F, 3.0F});
    expect_near(input_times_gradient.data(), {3.0F, 6.0F});
    EXPECT_FALSE(input.has_grad());
    EXPECT_FALSE(model.weight().has_grad());
    EXPECT_FALSE(model.bias().has_grad());
}

TEST(Explain, IntegratedGradientsIsExactAndCompleteForLinearModel) {
    auto model = linear_model();
    const clnn::Tensor input({1.5F, -2.0F}, {1, 2});
    const auto baseline = clnn::Tensor::zeros(input.shape());

    const auto attribution =
        clnn::explain::integrated_gradients(model, input, 0, baseline, 16);
    expect_near(attribution.data(), {3.0F, 6.0F});

    const auto input_output = model(input).data()[0];
    const auto baseline_output = model(baseline).data()[0];
    const auto attribution_sum = attribution.data()[0] + attribution.data()[1];
    EXPECT_NEAR(attribution_sum, input_output - baseline_output, 1.0e-5F);
}

TEST(Explain, BatchedTargetCanSelectAllSamplesOrOneSample) {
    auto model = linear_model();
    const clnn::Tensor input({1.0F, 2.0F, -1.0F, 3.0F}, {2, 2});

    expect_near(clnn::explain::saliency(model, input, 1).data(),
                {1.0F, 4.0F, 1.0F, 4.0F});
    expect_near(clnn::explain::saliency(model, input, {1, 1}).data(),
                {0.0F, 0.0F, 1.0F, 4.0F});
}

TEST(Explain, SmoothGradIsDeterministicAndPreservesShape) {
    std::mt19937 generator(9);
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::Linear>(2, 3, generator))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::Linear>(3, 2, generator));
    const clnn::Tensor input({0.1F, -0.2F, 0.7F, 0.4F}, {2, 2});

    const auto first = clnn::explain::smoothgrad(model, input, 1, 12, 0.2F, 42);
    const auto second = clnn::explain::smoothgrad(model, input, 1, 12, 0.2F, 42);

    EXPECT_EQ(first.shape(), input.shape());
    expect_near(first.data(), second.data(), 0.0F);
}

TEST(Explain, ValidatesTargetsBaselinesAndCounts) {
    auto model = linear_model();
    const clnn::Tensor input({1.0F, 2.0F}, {1, 2});

    EXPECT_THROW(static_cast<void>(clnn::explain::saliency(model, input, 2)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(clnn::explain::saliency(model, input, {0, 1})),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(clnn::explain::integrated_gradients(
                     model, input, 0, clnn::Tensor::zeros({2}), 10)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::explain::integrated_gradients(
                     model, input, 0, clnn::Tensor::zeros(input.shape()), 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::explain::smoothgrad(model, input, 0, 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::explain::smoothgrad(model, input, 0, 2, -0.1F)),
                 std::invalid_argument);
}

TEST(Explain, PreservesParametersExistingGradientsAndTrainingModeAcrossCalls) {
    auto model = linear_model();
    model.train();
    clnn::Tensor training_input({1.0F, 2.0F}, {1, 2}, true);
    clnn::sum(model(training_input)).backward();
    const auto weight_values = model.weight().data();
    const auto bias_values = model.bias().data();
    const auto weight_gradient = model.weight().grad();
    const auto bias_gradient = model.bias().grad();
    const clnn::Tensor explained({-2.0F, 0.5F}, {1, 2});

    const auto first = clnn::explain::saliency(model, explained, 0);
    const auto second = clnn::explain::saliency(model, explained, 0);

    expect_near(first.data(), second.data(), 0.0F);
    expect_near(model.weight().data(), weight_values, 0.0F);
    expect_near(model.bias().data(), bias_values, 0.0F);
    expect_near(model.weight().grad(), weight_gradient, 0.0F);
    expect_near(model.bias().grad(), bias_gradient, 0.0F);
    EXPECT_TRUE(model.training());
}

TEST(Explain, EnablesGradientsTemporarilyInsideNoGradScope) {
    auto model = linear_model();
    const clnn::Tensor input({1.0F, 2.0F}, {1, 2});
    clnn::NoGradGuard no_grad;
    EXPECT_FALSE(clnn::grad_enabled());
    const auto attribution = clnn::explain::saliency(model, input, 0);
    expect_near(attribution.data(), {2.0F, 3.0F});
    EXPECT_FALSE(clnn::grad_enabled());
    EXPECT_FALSE(attribution.requires_grad());
}

TEST(ModuleHooks, ObserveNestedOutputsAndRemoveSafely) {
    std::mt19937 generator(2);
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::Linear>(2, 2, generator))
        .add(std::make_unique<clnn::nn::ReLU>());
    auto& linear = *model.modules()[0];
    std::size_t calls = 0;
    clnn::Tensor captured;
    auto handle = linear.register_forward_hook(
        [&](clnn::nn::Module& source, const clnn::Tensor& hook_input,
            const clnn::Tensor& output) {
            EXPECT_EQ(&source, &linear);
            EXPECT_EQ(hook_input.shape(), (clnn::Shape{1, 2}));
            captured = output;
            ++calls;
        });

    clnn::Tensor input({1.0F, -2.0F}, {1, 2}, true);
    clnn::sum(model(input)).backward();
    EXPECT_EQ(calls, 1U);
    EXPECT_TRUE(captured.has_grad());
    EXPECT_TRUE(handle.active());
    handle.remove();
    static_cast<void>(model(input));
    EXPECT_EQ(calls, 1U);
    EXPECT_FALSE(handle.active());
}

TEST(ModuleHooks, HandleOutlivingModuleBecomesInactive) {
    clnn::nn::ForwardHookHandle handle;
    {
        auto model = linear_model();
        handle = model.register_forward_hook(
            [](clnn::nn::Module&, const clnn::Tensor&, const clnn::Tensor&) {});
        EXPECT_TRUE(handle.active());
    }
    EXPECT_FALSE(handle.active());
    EXPECT_NO_THROW(handle.remove());
}

TEST(Explain, OpenCLMatchesCpuWhenAvailable) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    auto cpu_model = linear_model();
    auto gpu_model = linear_model(gpu);
    const clnn::Tensor input({1.0F, 2.0F, -1.0F, 3.0F}, {2, 2});
    const auto baseline = clnn::Tensor::zeros(input.shape());

    const auto cpu_saliency = clnn::explain::saliency(cpu_model, input, 1);
    const auto gpu_saliency = clnn::explain::saliency(gpu_model, input.to(gpu), 1);
    const auto cpu_ig =
        clnn::explain::integrated_gradients(cpu_model, input, 0, baseline, 8);
    const auto gpu_ig = clnn::explain::integrated_gradients(
        gpu_model, input.to(gpu), 0, baseline.to(gpu), 8);
    const auto cpu_input_gradient = clnn::explain::input_x_gradient(cpu_model, input, 1);
    const auto gpu_input_gradient =
        clnn::explain::input_x_gradient(gpu_model, input.to(gpu), 1);
    const auto cpu_smooth = clnn::explain::smoothgrad(cpu_model, input, 1, 4, 0.1F, 7);
    const auto gpu_smooth =
        clnn::explain::smoothgrad(gpu_model, input.to(gpu), 1, 4, 0.1F, 7);

    EXPECT_EQ(gpu_saliency.device(), gpu);
    EXPECT_EQ(gpu_ig.device(), gpu);
    expect_near(gpu_saliency.data(), cpu_saliency.data(), 1.0e-5F);
    expect_near(gpu_ig.data(), cpu_ig.data(), 1.0e-5F);
    expect_near(gpu_input_gradient.data(), cpu_input_gradient.data(), 1.0e-5F);
    expect_near(gpu_smooth.data(), cpu_smooth.data(), 1.0e-5F);
}

} // namespace
