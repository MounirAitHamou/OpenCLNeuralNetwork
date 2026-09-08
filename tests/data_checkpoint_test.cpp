#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <random>

#ifndef CLNN_DATA_DIR
#error CLNN_DATA_DIR must be defined for data tests
#endif

TEST(Data, LoadsSplitsAndBatchesCsv) {
    const auto dataset = clnn::data::load_csv_numerical(std::filesystem::path(CLNN_DATA_DIR) /
                                                            "XOR" / "xor_data.csv",
                                                        {"bit1", "bit2"}, {"outputbit"});
    ASSERT_EQ(dataset.size(), 4U);
    EXPECT_EQ(dataset.input_shape(), (clnn::Shape{2}));
    auto partitions = clnn::data::split(dataset, 0.5F, 0.25F, 42);
    EXPECT_EQ(partitions.training.size(), 2U);
    EXPECT_EQ(partitions.validation.size(), 1U);
    EXPECT_EQ(partitions.test.size(), 1U);
    clnn::data::DataLoader loader(dataset, 3, clnn::Device::cpu(), false);
    auto first = loader.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->inputs.shape(), (clnn::Shape{3, 2}));
    EXPECT_EQ(first->targets.shape(), (clnn::Shape{3, 1}));
    ASSERT_TRUE(loader.next().has_value());
    EXPECT_FALSE(loader.next().has_value());
}

TEST(Checkpoint, RestoresArchitectureAndParameters) {
    std::mt19937 generator(13);
    clnn::nn::Sequential model;
    model
        .add(std::make_unique<clnn::nn::Conv2d>(1, 2, std::array<std::size_t, 2>{3, 3}, generator,
                                                std::array<std::size_t, 2>{1, 1},
                                                clnn::nn::PaddingMode::same))
        .add(std::make_unique<clnn::nn::LeakyReLU>(0.2F))
        .add(std::make_unique<clnn::nn::Flatten>())
        .add(std::make_unique<clnn::nn::Linear>(18, 2, generator))
        .add(std::make_unique<clnn::nn::Softmax>(1));
    const clnn::Tensor input({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
    const auto expected = model(input).data();
    const auto path = std::filesystem::temp_directory_path() / "clnn_full_checkpoint.bin";
    clnn::nn::save_checkpoint(model, path);
    auto restored = clnn::nn::load_checkpoint(path);
    std::filesystem::remove(path);
    EXPECT_EQ(restored->size(), model.size());
    EXPECT_EQ((*restored)(input).data(), expected);
}

TEST(Checkpoint, RestoresOptimizerMomentsForExactContinuation) {
    clnn::Tensor first({1.0F, -2.0F}, {2}, true);
    clnn::optim::Adam optimizer({&first}, 0.01F);
    clnn::sum(first * first).backward();
    optimizer.step();
    optimizer.zero_grad();

    const auto path = std::filesystem::temp_directory_path() / "clnn_optimizer_state.bin";
    clnn::optim::save_state_dict(optimizer, path);
    clnn::Tensor second(first.data(), first.shape(), true);
    clnn::optim::Adam restored({&second}, 0.5F);
    clnn::optim::load_state_dict(restored, path);
    std::filesystem::remove(path);

    clnn::sum(first * first).backward();
    optimizer.step();
    clnn::sum(second * second).backward();
    restored.step();
    EXPECT_EQ(first.data(), second.data());
}

TEST(Inspection, ReportsTensorHealthAndValues) {
    const clnn::Tensor tensor({-1, 0, 2, 3}, {2, 2});
    const auto stats = clnn::statistics(tensor);
    EXPECT_FLOAT_EQ(stats.minimum, -1.0F);
    EXPECT_FLOAT_EQ(stats.maximum, 3.0F);
    EXPECT_FLOAT_EQ(stats.mean, 1.0F);
    EXPECT_EQ(stats.non_finite, 0U);
    EXPECT_NE(clnn::inspect(tensor).find("shape=[2, 2]"), std::string::npos);
}
