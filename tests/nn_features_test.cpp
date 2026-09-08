#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

namespace {

void expect_values_near(const std::vector<float>& actual, const std::vector<float>& expected,
                        const float tolerance = 1.0e-4F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
}

TEST(AxisReduction, ComputesShapesValuesAndGradients) {
    clnn::Tensor input({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    auto reduced = clnn::sum(input, 0);
    EXPECT_EQ(reduced.shape(), (clnn::Shape{3}));
    EXPECT_EQ(reduced.data(), (std::vector<float>{5, 7, 9}));
    reduced.backward(clnn::Tensor({1, 2, 3}, {3}));
    EXPECT_EQ(input.grad(), (std::vector<float>{1, 2, 3, 1, 2, 3}));

    clnn::Tensor kept({1, 2, 3, 4, 5, 6}, {2, 3}, true);
    auto averaged = clnn::mean(kept, 1, true);
    EXPECT_EQ(averaged.shape(), (clnn::Shape{2, 1}));
    EXPECT_EQ(averaged.data(), (std::vector<float>{2, 5}));
    clnn::sum(averaged).backward();
    expect_values_near(kept.grad(), std::vector<float>(6, 1.0F / 3.0F));

    EXPECT_THROW(static_cast<void>(clnn::sum(kept, 2)), std::invalid_argument);
}

TEST(Pooling, ComputesMaxAndAverageForwardAndBackward) {
    const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9};
    clnn::Tensor maximum_input(values, {1, 1, 3, 3}, true);
    auto maximum = clnn::max_pool2d(maximum_input, {2, 2}, {1, 1});
    EXPECT_EQ(maximum.shape(), (clnn::Shape{1, 1, 2, 2}));
    EXPECT_EQ(maximum.data(), (std::vector<float>{5, 6, 8, 9}));
    clnn::sum(maximum).backward();
    EXPECT_EQ(maximum_input.grad(), (std::vector<float>{0, 0, 0, 0, 1, 1, 0, 1, 1}));

    clnn::Tensor average_input(values, {1, 1, 3, 3}, true);
    auto average = clnn::avg_pool2d(average_input, {2, 2}, {1, 1});
    EXPECT_EQ(average.data(), (std::vector<float>{3, 4, 6, 7}));
    clnn::sum(average).backward();
    expect_values_near(average_input.grad(),
                       {0.25F, 0.5F, 0.25F, 0.5F, 1.0F, 0.5F, 0.25F, 0.5F, 0.25F});
}

TEST(ModuleMode, PropagatesAndControlsDropout) {
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::Dropout>(0.5F, 7));
    const clnn::Tensor input(std::vector<float>(64, 1.0F), {8, 8});
    const auto training_output = model(input).data();
    EXPECT_TRUE(model.training());
    EXPECT_TRUE(model.modules().front()->training());
    EXPECT_NE(std::find(training_output.begin(), training_output.end(), 0.0F),
              training_output.end());
    for (const auto value : training_output)
        EXPECT_TRUE(value == 0.0F || value == 2.0F);

    model.eval();
    EXPECT_FALSE(model.training());
    EXPECT_FALSE(model.modules().front()->training());
    EXPECT_EQ(model(input).data(), input.data());
    model.train();
    EXPECT_TRUE(model.modules().front()->training());
}

TEST(BatchNormalization, NormalizesTracksStatisticsAndDifferentiates) {
    clnn::nn::BatchNorm normalization(2);
    clnn::Tensor input({1, 2, 3, 4, 5, 6}, {3, 2}, true);
    auto output = normalization(input);
    expect_values_near(output.data(), {-1.22474F, -1.22474F, 0, 0, 1.22474F, 1.22474F}, 2.0e-4F);
    output.backward(clnn::Tensor({1, -1, 2, -2, 3, -3}, {3, 2}));
    EXPECT_EQ(input.grad().size(), input.size());
    EXPECT_EQ(normalization.weight().grad().size(), 2U);
    expect_values_near(normalization.running_mean().data(), {0.3F, 0.4F});
    expect_values_near(normalization.running_variance().data(), {1.1666666F, 1.1666666F});

    normalization.eval();
    const auto evaluated = normalization(input.detach());
    EXPECT_EQ(evaluated.shape(), input.shape());
    for (const auto value : evaluated.data())
        EXPECT_TRUE(std::isfinite(value));
}

TEST(LayerNormalization, NormalizesTrailingDimensionsAndMatchesNumericalGradient) {
    clnn::nn::LayerNorm normalization({2, 2}, 1.0e-5F, false);
    const std::vector<float> values{1.0F, 2.0F, 4.0F, 8.0F, -3.0F, 0.0F, 3.0F, 6.0F};
    const std::vector<float> upstream{0.5F, -1.0F, 0.25F, 2.0F, -0.75F, 1.25F, -2.0F, 0.5F};
    clnn::Tensor input(values, {2, 2, 2}, true);
    auto output = normalization(input);
    output.backward(clnn::Tensor(upstream, output.shape()));

    const auto normalized = output.data();
    for (std::size_t batch = 0; batch < 2; ++batch) {
        float average = 0.0F;
        float square_average = 0.0F;
        for (std::size_t index = 0; index < 4; ++index) {
            const auto value = normalized[batch * 4 + index];
            average += value / 4.0F;
            square_average += value * value / 4.0F;
        }
        EXPECT_NEAR(average, 0.0F, 1.0e-5F);
        EXPECT_NEAR(square_average, 1.0F, 5.0e-5F);
    }

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t checked = 0; checked < values.size(); ++checked) {
        auto positive = values;
        auto negative = values;
        positive[checked] += epsilon;
        negative[checked] -= epsilon;
        const auto objective = [&](const std::vector<float>& candidate) {
            const auto result = normalization(clnn::Tensor(candidate, {2, 2, 2}));
            return clnn::sum(result * clnn::Tensor(upstream, result.shape())).item();
        };
        const auto numerical = (objective(positive) - objective(negative)) / (2.0F * epsilon);
        EXPECT_NEAR(input.grad()[checked], numerical, 8.0e-3F) << "at element " << checked;
    }
}

TEST(ModulePrimitives, GlobalAveragePoolingAndResidualCompose) {
    clnn::nn::GlobalAvgPool2d pooling;
    const clnn::Tensor image({1, 2, 3, 4, 10, 20, 30, 40}, {1, 2, 2, 2});
    EXPECT_EQ(pooling(image).shape(), (clnn::Shape{1, 2}));
    EXPECT_EQ(pooling(image).data(), (std::vector<float>{2.5F, 25.0F}));

    clnn::nn::Residual residual(std::make_unique<clnn::nn::ReLU>());
    const clnn::Tensor values({-2, -1, 0, 3}, {2, 2});
    EXPECT_EQ(residual(values).data(), (std::vector<float>{-2, -1, 0, 6}));
    EXPECT_EQ(residual.type(), clnn::nn::ModuleType::residual);
}

TEST(CheckpointFeatures, RestoresPoolingDropoutAndBatchNormalizationState) {
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::BatchNorm>(1))
        .add(std::make_unique<clnn::nn::MaxPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::AvgPool2d>(std::array<std::size_t, 2>{2, 2}))
        .add(std::make_unique<clnn::nn::Dropout>(0.25F, 9));
    const clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 1, 4, 4});
    static_cast<void>(model(input));
    model.eval();
    const auto expected = model(input).data();
    const auto path = std::filesystem::temp_directory_path() / "clnn_new_modules.clnn";
    clnn::nn::save_checkpoint(model, path);
    auto restored = clnn::nn::load_checkpoint(path);
    std::filesystem::remove(path);
    restored->eval();

    ASSERT_EQ(restored->size(), 4U);
    EXPECT_EQ(restored->modules()[0]->type(), clnn::nn::ModuleType::batch_normalization);
    EXPECT_EQ(restored->modules()[1]->type(), clnn::nn::ModuleType::max_pool_2d);
    EXPECT_EQ(restored->modules()[2]->type(), clnn::nn::ModuleType::average_pool_2d);
    EXPECT_EQ(restored->modules()[3]->type(), clnn::nn::ModuleType::dropout);
    expect_values_near((*restored)(input).data(), expected);
}

TEST(CheckpointFeatures, RestoresLayerNormalizationAndGlobalPooling) {
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::LayerNorm>(clnn::Shape{2, 2}))
        .add(std::make_unique<clnn::nn::GlobalAvgPool2d>());
    const clnn::Tensor input({1, 2, 4, 8, -3, 0, 3, 6}, {1, 2, 2, 2});
    const auto expected = model(input).data();
    const auto path = std::filesystem::temp_directory_path() / "clnn_normalization_modules.clnn";
    clnn::nn::save_checkpoint(model, path);
    auto restored = clnn::nn::load_checkpoint(path);
    std::filesystem::remove(path);

    ASSERT_EQ(restored->size(), 2U);
    EXPECT_EQ(restored->modules()[0]->type(), clnn::nn::ModuleType::layer_normalization);
    EXPECT_EQ(restored->modules()[1]->type(), clnn::nn::ModuleType::global_average_pool_2d);
    expect_values_near((*restored)(input).data(), expected);
}

TEST(OpenCLFeatures, PoolingReductionsBatchNormAndTiledMatmulMatchCpu) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    const std::vector<float> image_values{
        -1, 2, 3,  4, 5, -6, 7, 8,  9,  10, -11, 12, 13,  14, 15, -16,
        2,  1, -3, 4, 6, 5,  8, -7, 10, 9,  12,  11, -14, 13, 16, 15,
    };
    clnn::Tensor cpu_image(image_values, {2, 1, 4, 4}, true);
    clnn::Tensor gpu_image(image_values, {2, 1, 4, 4}, true, {}, gpu);
    auto cpu_output = clnn::mean(clnn::max_pool2d(cpu_image, {2, 2}), 0, true) +
                      clnn::avg_pool2d(cpu_image, {2, 2});
    auto gpu_output = clnn::mean(clnn::max_pool2d(gpu_image, {2, 2}), 0, true) +
                      clnn::avg_pool2d(gpu_image, {2, 2});
    clnn::sum(cpu_output).backward();
    clnn::sum(gpu_output).backward();

    clnn::nn::BatchNorm cpu_normalization(1);
    clnn::nn::BatchNorm gpu_normalization(1, 1.0e-5F, 0.1F, true, gpu);
    auto cpu_normalized = cpu_normalization(cpu_image.detach());
    auto gpu_normalized = gpu_normalization(gpu_image.detach());
    cpu_normalized.backward(clnn::Tensor(std::vector<float>(32, 0.25F), {2, 1, 4, 4}));
    gpu_normalized.backward(
        clnn::Tensor(std::vector<float>(32, 0.25F), {2, 1, 4, 4}, false, {}, gpu));

    std::vector<float> lhs_values(17 * 19), rhs_values(19 * 18);
    for (std::size_t index = 0; index < lhs_values.size(); ++index)
        lhs_values[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 7.0F;
    for (std::size_t index = 0; index < rhs_values.size(); ++index)
        rhs_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 9.0F;
    const clnn::Tensor cpu_lhs(lhs_values, {17, 19});
    const clnn::Tensor cpu_rhs(rhs_values, {19, 18});
    auto cpu_matrix = clnn::matmul(cpu_lhs, cpu_rhs);
    auto gpu_matrix = clnn::matmul(cpu_lhs.to(gpu), cpu_rhs.to(gpu));
    gpu.synchronize();

    expect_values_near(gpu_output.data(), cpu_output.data(), 1.0e-4F);
    expect_values_near(gpu_image.grad(), cpu_image.grad(), 1.0e-4F);
    expect_values_near(gpu_normalized.data(), cpu_normalized.data(), 3.0e-4F);
    expect_values_near(gpu_normalization.weight().grad(), cpu_normalization.weight().grad(),
                       3.0e-4F);
    expect_values_near(gpu_normalization.running_mean().data(),
                       cpu_normalization.running_mean().data(), 1.0e-4F);
    expect_values_near(gpu_matrix.data(), cpu_matrix.data(), 5.0e-4F);
}

TEST(OpenCLRuntime, ProfilesKernelCacheAndReusesCompletedBuffers) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const auto gpu = clnn::Device::opencl();
    gpu.synchronize();
    gpu.clear_memory_pool();
    gpu.set_profiling(true);
    static_cast<void>(gpu.profile());
    const clnn::Tensor input(std::vector<float>(4096, 0.25F), {64, 64}, false, {}, gpu);

    for (int iteration = 0; iteration < 4; ++iteration) {
        {
            auto result = clnn::relu(input + 1.0F);
            gpu.synchronize();
            EXPECT_FLOAT_EQ(result.data().front(), 1.25F);
        }
    }
    const auto profiles = gpu.profile();
    const auto statistics = gpu.runtime_statistics();
    gpu.set_profiling(false);

    EXPECT_FALSE(profiles.empty());
    EXPECT_TRUE(std::any_of(profiles.begin(), profiles.end(), [](const auto& profile) {
        return profile.operation == "binary_op" && profile.calls >= 4 &&
               profile.total_milliseconds >= 0.0;
    }));
    EXPECT_GT(statistics.cached_kernels, 0U);
    EXPECT_GT(statistics.buffer_reuses, 0U);
    EXPECT_GT(statistics.pooled_buffers, 0U);
}

} // namespace
