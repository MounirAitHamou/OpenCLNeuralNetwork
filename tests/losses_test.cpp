#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

TEST(Losses, MeanSquaredErrorHasExpectedValueAndGradient) {
    clnn::Tensor prediction({1, 2, 3, 4}, {2, 2}, true);
    const clnn::Tensor target({0, 2, 5, 3}, {2, 2});
    auto loss = clnn::mse_loss(prediction, target);
    EXPECT_FLOAT_EQ(loss.item(), 1.5F);
    loss.backward();
    const std::vector<float> expected{0.5F, 0.0F, -1.0F, 0.5F};
    EXPECT_EQ(prediction.grad(), expected);
}

TEST(Losses, CrossEntropyIsStableAndHasZeroSumGradientPerRow) {
    clnn::Tensor logits({1001, 1000, 999, -1000, -999, -998}, {2, 3}, true);
    auto loss = clnn::cross_entropy(logits, {0, 2});
    EXPECT_TRUE(std::isfinite(loss.item()));
    loss.backward();
    for (std::size_t row = 0; row < 2; ++row) {
        const auto begin = logits.grad().begin() + static_cast<std::ptrdiff_t>(row * 3);
        EXPECT_NEAR(std::accumulate(begin, begin + 3, 0.0F), 0.0F, 1.0e-6F);
    }
}

TEST(Losses, BinaryCrossEntropyWithLogitsIsStableAtExtremeValues) {
    clnn::Tensor logits({-100.0F, 100.0F, 0.0F}, {3}, true);
    const clnn::Tensor targets({0.0F, 1.0F, 1.0F}, {3});
    auto loss = clnn::binary_cross_entropy_with_logits(logits, targets);
    EXPECT_TRUE(std::isfinite(loss.item()));
    loss.backward();
    for (const auto gradient : logits.grad()) {
        EXPECT_TRUE(std::isfinite(gradient));
    }
}

TEST(Losses, RejectsInvalidTargets) {
    const clnn::Tensor logits({1, 2, 3, 4}, {2, 2});
    EXPECT_THROW(clnn::cross_entropy(logits, {0}), std::invalid_argument);
    EXPECT_THROW(clnn::cross_entropy(logits, {0, 2}), std::out_of_range);
}
