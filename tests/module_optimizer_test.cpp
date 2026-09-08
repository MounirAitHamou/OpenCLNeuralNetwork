#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <random>

TEST(Module, LinearOwnsDiscoverableParameters) {
    std::mt19937 generator(7);
    clnn::nn::Linear layer(3, 2, generator);
    const auto named = layer.named_parameters();
    ASSERT_EQ(named.size(), 2U);
    EXPECT_EQ(named[0].name, "weight");
    EXPECT_EQ(named[0].tensor->shape(), (clnn::Shape{3, 2}));
    EXPECT_EQ(named[1].name, "bias");
    EXPECT_EQ(named[1].tensor->shape(), (clnn::Shape{2}));
}

TEST(Optimizer, SGDTrainsLinearRegression) {
    std::mt19937 generator(11);
    clnn::nn::Linear model(1, 1, generator);
    clnn::optim::SGD optimizer(model.parameters(), 0.05F, 0.8F);
    const clnn::Tensor inputs({-2, -1, 0, 1, 2}, {5, 1});
    const clnn::Tensor targets({-3, -1, 1, 3, 5}, {5, 1});

    float initial_loss = 0.0F;
    float final_loss = 0.0F;
    for (int step = 0; step < 200; ++step) {
        optimizer.zero_grad();
        auto loss = clnn::mse_loss(model(inputs), targets);
        if (step == 0) {
            initial_loss = loss.item();
        }
        final_loss = loss.item();
        loss.backward();
        optimizer.step();
    }
    EXPECT_LT(final_loss, initial_loss * 1.0e-4F);
    EXPECT_NEAR(model.weight().data()[0], 2.0F, 1.0e-3F);
    EXPECT_NEAR(model.bias().data()[0], 1.0F, 1.0e-3F);
}

TEST(Optimizer, AdamWUpdatesAndClearsParameters) {
    clnn::Tensor parameter({1.0F, -1.0F}, {2}, true);
    clnn::optim::AdamW optimizer({&parameter}, 0.1F, 0.9F, 0.999F, 1.0e-8F, 0.01F);
    clnn::sum(parameter * parameter).backward();
    optimizer.step();
    EXPECT_LT(parameter.data()[0], 1.0F);
    EXPECT_GT(parameter.data()[1], -1.0F);
    optimizer.zero_grad();
    EXPECT_FALSE(parameter.has_grad());
}

TEST(Optimizer, ParameterGroupsApplyIndependentLearningRatesAndWeightDecay) {
    clnn::Tensor fast({1.0F}, {1}, true);
    clnn::Tensor slow({1.0F}, {1}, true);
    clnn::optim::SGD optimizer(
        clnn::optim::ParameterGroups{{{{&fast}, 0.1F, 0.0F}, {{&slow}, 0.01F, 1.0F}}});
    clnn::sum(fast * fast + slow * slow).backward();
    optimizer.step();

    EXPECT_EQ(optimizer.parameter_group_count(), 2U);
    EXPECT_NEAR(fast.data()[0], 0.8F, 1.0e-6F);
    EXPECT_NEAR(slow.data()[0], 0.97F, 1.0e-6F);
    EXPECT_FLOAT_EQ(optimizer.learning_rate(0), 0.1F);
    EXPECT_FLOAT_EQ(optimizer.learning_rate(1), 0.01F);
}

TEST(Optimizer, LearningRateSchedulersUpdateEveryParameterGroup) {
    clnn::Tensor first({1.0F}, {1}, true);
    clnn::Tensor second({1.0F}, {1}, true);
    clnn::optim::SGD optimizer(
        clnn::optim::ParameterGroups{{{{&first}, 0.1F}, {{&second}, 0.02F}}});

    clnn::optim::StepLR step_scheduler(optimizer, 2, 0.5F);
    step_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer.learning_rate(0), 0.1F);
    step_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer.learning_rate(0), 0.05F);
    EXPECT_FLOAT_EQ(optimizer.learning_rate(1), 0.01F);

    clnn::optim::ExponentialLR exponential(optimizer, 0.5F);
    exponential.step();
    EXPECT_FLOAT_EQ(optimizer.learning_rate(0), 0.025F);
    EXPECT_FLOAT_EQ(optimizer.learning_rate(1), 0.005F);

    clnn::optim::CosineAnnealingLR cosine(optimizer, 2);
    cosine.step();
    EXPECT_NEAR(optimizer.learning_rate(0), 0.0125F, 1.0e-7F);
    cosine.step();
    EXPECT_FLOAT_EQ(optimizer.learning_rate(0), 0.0F);
    EXPECT_EQ(cosine.steps(), 2U);
}

TEST(Optimizer, ParameterGroupSettingsRoundTripWithState) {
    clnn::Tensor source_first({1.0F}, {1}, true);
    clnn::Tensor source_second({2.0F}, {1}, true);
    clnn::optim::SGD source(
        clnn::optim::ParameterGroups{{{{&source_first}, 0.1F}, {{&source_second}, 0.02F, 0.3F}}},
        0.5F);
    const auto path = std::filesystem::temp_directory_path() / "clnn_parameter_groups.bin";
    clnn::optim::save_state_dict(source, path);

    clnn::Tensor restored_first({1.0F}, {1}, true);
    clnn::Tensor restored_second({2.0F}, {1}, true);
    clnn::optim::SGD restored(
        clnn::optim::ParameterGroups{{{{&restored_first}, 0.5F}, {{&restored_second}, 0.6F}}},
        0.5F);
    clnn::optim::load_state_dict(restored, path);
    std::filesystem::remove(path);

    EXPECT_FLOAT_EQ(restored.learning_rate(0), 0.1F);
    EXPECT_FLOAT_EQ(restored.learning_rate(1), 0.02F);
}

TEST(Module, StateDictRoundTripsByNameAndShape) {
    std::mt19937 first_generator(1);
    std::mt19937 second_generator(2);
    clnn::nn::Sequential original;
    original.add(std::make_unique<clnn::nn::Linear>(2, 3, first_generator))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::Linear>(3, 1, first_generator));
    clnn::nn::Sequential restored;
    restored.add(std::make_unique<clnn::nn::Linear>(2, 3, second_generator))
        .add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::Linear>(3, 1, second_generator));

    const auto path = std::filesystem::temp_directory_path() / "clnn_state_round_trip.bin";
    clnn::nn::save_state_dict(original, path);
    clnn::nn::load_state_dict(restored, path);
    std::filesystem::remove(path);

    const clnn::Tensor input({1, 2, 3, 4}, {2, 2});
    EXPECT_EQ(original(input).data(), restored(input).data());
}

TEST(Module, SequentialTrainsEndToEnd) {
    std::mt19937 generator(42);
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::Linear>(2, 6, generator))
        .add(std::make_unique<clnn::nn::Tanh>())
        .add(std::make_unique<clnn::nn::Linear>(6, 1, generator));
    clnn::optim::Adam optimizer(model.parameters(), 0.03F);
    const clnn::Tensor inputs({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    const clnn::Tensor targets({0, 1, 1, 0}, {4, 1});

    float final_loss = 0.0F;
    for (int step = 0; step < 800; ++step) {
        optimizer.zero_grad();
        auto loss = clnn::binary_cross_entropy_with_logits(model(inputs), targets);
        final_loss = loss.item();
        loss.backward();
        optimizer.step();
    }
    EXPECT_LT(final_loss, 0.03F);
}
