#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

class TemporaryFile final {
  public:
    explicit TemporaryFile(std::string name)
        : path_(std::filesystem::temp_directory_path() / ("clnn_" + std::move(name))) {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

    void text(const std::string& contents) const {
        std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream);
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }

  private:
    std::filesystem::path path_;
};

template <typename Value> void write_value(std::ofstream& stream, const Value& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

class IdentityModule final : public clnn::nn::Module {
  public:
    clnn::Tensor forward(const clnn::Tensor& input) override {
        return input;
    }
};

class NamedModule final : public clnn::nn::Module {
  public:
    NamedModule(std::string name, clnn::Shape shape)
        : name_(std::move(name)), parameter_(clnn::Tensor::ones(shape, true)) {}
    clnn::Tensor forward(const clnn::Tensor& input) override {
        return input;
    }
    std::vector<clnn::nn::NamedParameter> named_parameters() override {
        return {{name_, &parameter_}};
    }

  private:
    std::string name_;
    clnn::Tensor parameter_;
};

class DuplicateParameterModule final : public clnn::nn::Module {
  public:
    DuplicateParameterModule()
        : first_(clnn::Tensor::ones({1}, true)), second_(clnn::Tensor::zeros({1}, true)) {}
    clnn::Tensor forward(const clnn::Tensor& input) override {
        return input;
    }
    std::vector<clnn::nn::NamedParameter> named_parameters() override {
        return {{"duplicate", &first_}, {"duplicate", &second_}};
    }

  private:
    clnn::Tensor first_;
    clnn::Tensor second_;
};

class InvalidParameterModule final : public clnn::nn::Module {
  public:
    clnn::Tensor forward(const clnn::Tensor& input) override {
        return input;
    }
    std::vector<clnn::nn::NamedParameter> named_parameters() override {
        return {{"invalid", nullptr}};
    }
};

class LongNameModule final : public clnn::nn::Module {
  public:
    LongNameModule() : parameter_(clnn::Tensor::ones({1}, true)) {}
    clnn::Tensor forward(const clnn::Tensor& input) override {
        return input;
    }
    std::vector<clnn::nn::NamedParameter> named_parameters() override {
        return {{std::string((1U << 20U) + 1U, 'x'), &parameter_}};
    }

  private:
    clnn::Tensor parameter_;
};

class UnsupportedOptimizer final : public clnn::optim::Optimizer {
  public:
    explicit UnsupportedOptimizer(std::vector<clnn::Tensor*> parameters)
        : Optimizer(std::move(parameters)) {}
    void step() override {}
};

TEST(ModuleEdge, BaseModuleAndAllActivationModulesExposeTheirContracts) {
    IdentityModule identity;
    const clnn::Tensor input({-1, 2}, {1, 2});
    EXPECT_EQ(identity(input).data(), input.data());
    EXPECT_TRUE(identity.parameters().empty());
    EXPECT_TRUE(identity.named_parameters().empty());
    EXPECT_EQ(identity.type(), clnn::nn::ModuleType::custom);
    EXPECT_EQ(&identity.to(clnn::Device::cpu()), &identity);

    clnn::nn::ReLU relu;
    clnn::nn::Sigmoid sigmoid;
    clnn::nn::Tanh tanh;
    clnn::nn::LeakyReLU leaky(0.25F);
    clnn::nn::Softmax softmax(1);
    EXPECT_EQ(relu.type(), clnn::nn::ModuleType::relu);
    EXPECT_EQ(sigmoid.type(), clnn::nn::ModuleType::sigmoid);
    EXPECT_EQ(tanh.type(), clnn::nn::ModuleType::tanh);
    EXPECT_EQ(leaky.type(), clnn::nn::ModuleType::leaky_relu);
    EXPECT_EQ(softmax.type(), clnn::nn::ModuleType::softmax);
    EXPECT_FLOAT_EQ(leaky.negative_slope(), 0.25F);
    EXPECT_EQ(softmax.dimension(), 1U);
    EXPECT_EQ(relu(input).data(), (std::vector<float>{0, 2}));
    EXPECT_EQ(sigmoid(input).shape(), input.shape());
    EXPECT_EQ(tanh(input).shape(), input.shape());
    EXPECT_EQ(leaky(input).data(), (std::vector<float>{-0.25F, 2.0F}));
    EXPECT_THROW((clnn::nn::LeakyReLU(-0.1F)), std::invalid_argument);
}

TEST(ModuleEdge, LinearWithoutBiasAndFlattenValidateInputs) {
    std::mt19937 generator(5);
    clnn::nn::Linear linear(2, 3, generator, false);
    EXPECT_FALSE(linear.has_bias());
    EXPECT_EQ(linear.input_features(), 2U);
    EXPECT_EQ(linear.output_features(), 3U);
    EXPECT_EQ(linear.type(), clnn::nn::ModuleType::linear);
    EXPECT_EQ(linear.named_parameters().size(), 1U);
    EXPECT_THROW(static_cast<void>(linear.bias()), std::logic_error);
    EXPECT_THROW(static_cast<void>(linear(clnn::Tensor::ones({2}))), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(linear(clnn::Tensor::ones({1, 3}))), std::invalid_argument);
    EXPECT_THROW((clnn::nn::Linear(0, 1, generator)), std::invalid_argument);

    clnn::nn::Flatten flatten;
    EXPECT_EQ(flatten.type(), clnn::nn::ModuleType::flatten);
    EXPECT_EQ(flatten(clnn::Tensor::ones({2, 3, 4})).shape(), (clnn::Shape{2, 12}));
    EXPECT_THROW(static_cast<void>(flatten(clnn::Tensor::ones({3}))), std::invalid_argument);
}

TEST(ModuleEdge, ConvolutionMetadataBiasAndPaddingValidation) {
    std::mt19937 generator(8);
    clnn::nn::Conv2d convolution(2, 3, {3, 5}, generator, {2, 1}, {1, 2}, false);
    EXPECT_EQ(convolution.input_channels(), 2U);
    EXPECT_EQ(convolution.output_channels(), 3U);
    EXPECT_EQ(convolution.kernel_size(), (std::array<std::size_t, 2>{3, 5}));
    EXPECT_EQ(convolution.stride(), (std::array<std::size_t, 2>{2, 1}));
    EXPECT_EQ(convolution.padding(), (std::array<std::size_t, 2>{1, 2}));
    EXPECT_FALSE(convolution.has_bias());
    EXPECT_EQ(convolution.named_parameters().size(), 1U);
    EXPECT_EQ(convolution.type(), clnn::nn::ModuleType::convolution_2d);
    EXPECT_THROW(static_cast<void>(convolution.bias()), std::logic_error);
    EXPECT_THROW((clnn::nn::Conv2d(1, 1, {2, 2}, generator, {1, 1}, clnn::nn::PaddingMode::same)),
                 std::invalid_argument);
    EXPECT_THROW((clnn::nn::Conv2d(1, 1, {3, 3}, generator, {0, 1}, {0, 0})),
                 std::invalid_argument);

    clnn::nn::Conv2d same(1, 1, {3, 3}, generator, {1, 1}, clnn::nn::PaddingMode::same);
    EXPECT_EQ(same.padding(), (std::array<std::size_t, 2>{1, 1}));
    EXPECT_TRUE(same.has_bias());
    EXPECT_EQ(same.bias().shape(), (clnn::Shape{1}));
}

TEST(ModuleEdge, SequentialRejectsNullAndSupportsEmptyIdentity) {
    std::vector<std::unique_ptr<clnn::nn::Module>> invalid;
    invalid.push_back(nullptr);
    EXPECT_THROW((clnn::nn::Sequential(std::move(invalid))), std::invalid_argument);
    clnn::nn::Sequential sequential;
    EXPECT_THROW(sequential.add(nullptr), std::invalid_argument);
    const clnn::Tensor input({1, 2}, {2});
    EXPECT_EQ(sequential(input).data(), input.data());
    EXPECT_TRUE(sequential.modules().empty());
}

TEST(InspectionEdge, HandlesNonFiniteValuesAndTruncatedDisplay) {
    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const clnn::Tensor mixed({-2, infinity, nan, 4}, {4});
    const auto mixed_stats = clnn::statistics(mixed);
    EXPECT_FLOAT_EQ(mixed_stats.minimum, -2.0F);
    EXPECT_FLOAT_EQ(mixed_stats.maximum, 4.0F);
    EXPECT_FLOAT_EQ(mixed_stats.mean, 1.0F);
    EXPECT_EQ(mixed_stats.non_finite, 2U);
    EXPECT_NE(clnn::inspect(mixed, 2).find("..."), std::string::npos);

    const clnn::Tensor non_finite({infinity, nan}, {2});
    const auto stats = clnn::statistics(non_finite);
    EXPECT_TRUE(std::isnan(stats.minimum));
    EXPECT_TRUE(std::isnan(stats.maximum));
    EXPECT_TRUE(std::isnan(stats.mean));
    EXPECT_TRUE(std::isnan(stats.standard_deviation));
    EXPECT_EQ(stats.non_finite, 2U);
}

TEST(CheckpointEdge, RoundTripsEveryParameterlessActivationType) {
    clnn::nn::Sequential model;
    model.add(std::make_unique<clnn::nn::ReLU>())
        .add(std::make_unique<clnn::nn::Sigmoid>())
        .add(std::make_unique<clnn::nn::Tanh>());
    const TemporaryFile file("activation-checkpoint.bin");
    clnn::nn::save_checkpoint(model, file.path());
    auto restored = clnn::nn::load_checkpoint(file.path());
    ASSERT_EQ(restored->size(), 3U);
    EXPECT_EQ(restored->modules()[0]->type(), clnn::nn::ModuleType::relu);
    EXPECT_EQ(restored->modules()[1]->type(), clnn::nn::ModuleType::sigmoid);
    EXPECT_EQ(restored->modules()[2]->type(), clnn::nn::ModuleType::tanh);
    const clnn::Tensor input({-1, 2}, {1, 2});
    EXPECT_EQ((*restored)(input).data(), model(input).data());
}

TEST(StateDictEdge, RejectsInvalidDestinationsAndCorruptFiles) {
    std::mt19937 generator(12);
    clnn::nn::Linear with_bias(2, 1, generator);
    clnn::nn::Linear without_bias(2, 1, generator, false);
    const TemporaryFile valid("state-valid.bin");
    clnn::nn::save_state_dict(with_bias, valid.path());
    EXPECT_THROW(clnn::nn::load_state_dict(without_bias, valid.path()), std::runtime_error);

    const TemporaryFile shape("state-shape.bin");
    clnn::nn::save_state_dict(without_bias, shape.path());
    clnn::nn::Linear wrong_shape(3, 1, generator, false);
    EXPECT_THROW(clnn::nn::load_state_dict(wrong_shape, shape.path()), std::runtime_error);
    NamedModule renamed("renamed", {2, 1});
    EXPECT_THROW(clnn::nn::load_state_dict(renamed, shape.path()), std::runtime_error);

    DuplicateParameterModule duplicate;
    const TemporaryFile duplicate_file("state-duplicate.bin");
    clnn::nn::save_state_dict(duplicate, duplicate_file.path());
    EXPECT_THROW(clnn::nn::load_state_dict(duplicate, duplicate_file.path()), std::runtime_error);

    InvalidParameterModule invalid;
    EXPECT_THROW(clnn::nn::save_state_dict(invalid, valid.path()), std::logic_error);
    LongNameModule long_name;
    EXPECT_THROW(clnn::nn::save_state_dict(long_name, valid.path()), std::length_error);
    EXPECT_THROW(clnn::nn::save_state_dict(with_bias, std::filesystem::temp_directory_path()),
                 std::runtime_error);

    const TemporaryFile bad_magic("state-bad-magic.bin");
    bad_magic.text("not a state file");
    EXPECT_THROW(clnn::nn::load_state_dict(with_bias, bad_magic.path()), std::runtime_error);
    const TemporaryFile truncated("state-truncated.bin");
    truncated.text("CLNNAUT1");
    EXPECT_THROW(clnn::nn::load_state_dict(with_bias, truncated.path()), std::runtime_error);
    EXPECT_THROW(clnn::nn::load_state_dict(with_bias, TemporaryFile("missing-state.bin").path()),
                 std::runtime_error);
}

TEST(StateDictEdge, RejectsUnreasonableCountsRanksAndElementCounts) {
    std::mt19937 generator(2);
    clnn::nn::Linear model(1, 1, generator, false);

    const TemporaryFile count_file("state-count.bin");
    {
        std::ofstream stream(count_file.path(), std::ios::binary);
        stream.write("CLNNAUT1", 8);
        write_value(stream, std::uint64_t{1'000'001});
    }
    EXPECT_THROW(clnn::nn::load_state_dict(model, count_file.path()), std::runtime_error);

    const TemporaryFile rank_file("state-rank.bin");
    {
        std::ofstream stream(rank_file.path(), std::ios::binary);
        stream.write("CLNNAUT1", 8);
        write_value(stream, std::uint64_t{1});
        write_value(stream, std::uint32_t{1});
        stream.write("x", 1);
        write_value(stream, std::uint32_t{33});
    }
    EXPECT_THROW(clnn::nn::load_state_dict(model, rank_file.path()), std::runtime_error);

    const TemporaryFile elements_file("state-elements.bin");
    {
        std::ofstream stream(elements_file.path(), std::ios::binary);
        stream.write("CLNNAUT1", 8);
        write_value(stream, std::uint64_t{1});
        write_value(stream, std::uint32_t{6});
        stream.write("weight", 6);
        write_value(stream, std::uint32_t{1});
        write_value(stream, std::uint64_t{1});
        write_value(stream, std::uint64_t{2});
    }
    EXPECT_THROW(clnn::nn::load_state_dict(model, elements_file.path()), std::runtime_error);
}

TEST(CheckpointEdge, RejectsUnsupportedArchitecturesAndCorruptHeaders) {
    clnn::nn::Sequential custom;
    custom.add(std::make_unique<IdentityModule>());
    const TemporaryFile output("custom-checkpoint.bin");
    EXPECT_THROW(clnn::nn::save_checkpoint(custom, output.path()), std::runtime_error);
    EXPECT_THROW(clnn::nn::save_checkpoint(custom, std::filesystem::temp_directory_path()),
                 std::runtime_error);

    const TemporaryFile bad_magic("checkpoint-bad-magic.bin");
    bad_magic.text("not a checkpoint");
    EXPECT_THROW(static_cast<void>(clnn::nn::load_checkpoint(bad_magic.path())),
                 std::runtime_error);
    EXPECT_THROW(static_cast<void>(
                     clnn::nn::load_checkpoint(TemporaryFile("missing-checkpoint.bin").path())),
                 std::runtime_error);

    const TemporaryFile unreasonable("checkpoint-count.bin");
    {
        std::ofstream stream(unreasonable.path(), std::ios::binary);
        stream.write("CLNNCHK2", 8);
        write_value(stream, std::uint64_t{100'001});
    }
    EXPECT_THROW(static_cast<void>(clnn::nn::load_checkpoint(unreasonable.path())),
                 std::runtime_error);

    const TemporaryFile unsupported("checkpoint-unsupported.bin");
    {
        std::ofstream stream(unsupported.path(), std::ios::binary);
        stream.write("CLNNCHK2", 8);
        write_value(stream, std::uint64_t{1});
        write_value(stream, std::uint32_t{0});
    }
    EXPECT_THROW(static_cast<void>(clnn::nn::load_checkpoint(unsupported.path())),
                 std::runtime_error);
}

TEST(OptimizerEdge, RejectsInvalidParametersAndHyperparameters) {
    clnn::Tensor undefined;
    EXPECT_THROW((clnn::optim::SGD({nullptr}, 0.1F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::SGD({&undefined}, 0.1F)), std::invalid_argument);
    clnn::Tensor constant({1}, {1});
    EXPECT_THROW((clnn::optim::SGD({&constant}, 0.1F)), std::invalid_argument);
    clnn::Tensor leaf({1}, {1}, true);
    auto non_leaf = leaf * leaf;
    EXPECT_THROW((clnn::optim::SGD({&non_leaf}, 0.1F)), std::invalid_argument);

    EXPECT_THROW((clnn::optim::SGD({&leaf}, 0.0F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::SGD({&leaf}, 0.1F, -0.1F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::SGD({&leaf}, 0.1F, 1.0F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::SGD({&leaf}, 0.1F, 0.0F, -0.1F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::Adam({&leaf}, 0.0F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::Adam({&leaf}, 0.1F, -0.1F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::Adam({&leaf}, 0.1F, 0.9F, 1.0F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::Adam({&leaf}, 0.1F, 0.9F, 0.99F, 0.0F)), std::invalid_argument);
    EXPECT_THROW((clnn::optim::Adam({&leaf}, 0.1F, 0.9F, 0.99F, 1.0e-8F, -1.0F)),
                 std::invalid_argument);
}

TEST(OptimizerEdge, SgdWeightDecayNoMomentumAndMissingGradients) {
    clnn::Tensor parameter({2.0F}, {1}, true);
    clnn::optim::SGD optimizer({&parameter}, 0.1F, 0.0F, 0.5F);
    optimizer.step();
    EXPECT_FLOAT_EQ(parameter.data()[0], 2.0F);
    clnn::detail_accumulate(parameter, {3.0F});
    optimizer.step();
    EXPECT_NEAR(parameter.data()[0], 1.6F, 1.0e-6F);
    optimizer.zero_grad();
    EXPECT_FALSE(parameter.has_grad());
}

TEST(OptimizerEdge, SgdAndAdamStatesResumeExactly) {
    clnn::Tensor first_sgd({1.0F}, {1}, true);
    clnn::optim::SGD sgd({&first_sgd}, 0.1F, 0.5F, 0.1F);
    clnn::sum(first_sgd * first_sgd).backward();
    sgd.step();
    sgd.zero_grad();
    const TemporaryFile sgd_file("sgd-state.bin");
    clnn::optim::save_state_dict(sgd, sgd_file.path());
    clnn::Tensor second_sgd(first_sgd.data(), {1}, true);
    clnn::optim::SGD restored_sgd({&second_sgd}, 0.9F);
    clnn::optim::load_state_dict(restored_sgd, sgd_file.path());
    clnn::sum(first_sgd * first_sgd).backward();
    clnn::sum(second_sgd * second_sgd).backward();
    sgd.step();
    restored_sgd.step();
    EXPECT_EQ(first_sgd.data(), second_sgd.data());

    clnn::Tensor first_adam({1.0F, -2.0F}, {2}, true);
    clnn::optim::Adam adam({&first_adam}, 0.02F, 0.8F, 0.9F, 1.0e-6F, 0.1F, false);
    clnn::sum(first_adam * first_adam).backward();
    adam.step();
    adam.zero_grad();
    const TemporaryFile adam_file("adam-state-edge.bin");
    clnn::optim::save_state_dict(adam, adam_file.path());
    clnn::Tensor second_adam(first_adam.data(), {2}, true);
    clnn::optim::Adam restored_adam({&second_adam}, 0.5F);
    clnn::optim::load_state_dict(restored_adam, adam_file.path());
    clnn::sum(first_adam * first_adam).backward();
    clnn::sum(second_adam * second_adam).backward();
    adam.step();
    restored_adam.step();
    EXPECT_EQ(first_adam.data(), second_adam.data());
}

TEST(OptimizerEdge, StateIoRejectsUnsupportedMismatchedAndCorruptData) {
    clnn::Tensor parameter({1.0F}, {1}, true);
    UnsupportedOptimizer unsupported({&parameter});
    const TemporaryFile unsupported_file("unsupported-optimizer.bin");
    EXPECT_THROW(clnn::optim::save_state_dict(unsupported, unsupported_file.path()),
                 std::runtime_error);
    EXPECT_THROW(clnn::optim::save_state_dict(unsupported, std::filesystem::temp_directory_path()),
                 std::runtime_error);

    clnn::optim::SGD sgd({&parameter}, 0.1F);
    const TemporaryFile state("optimizer-valid.bin");
    clnn::optim::save_state_dict(sgd, state.path());
    clnn::optim::Adam wrong_type({&parameter});
    EXPECT_THROW(clnn::optim::load_state_dict(wrong_type, state.path()), std::runtime_error);
    clnn::optim::SGD wrong_count({}, 0.1F);
    EXPECT_THROW(clnn::optim::load_state_dict(wrong_count, state.path()), std::runtime_error);
    clnn::Tensor wrong_shape_parameter({1, 2}, {2}, true);
    clnn::optim::SGD wrong_shape({&wrong_shape_parameter}, 0.1F);
    EXPECT_THROW(clnn::optim::load_state_dict(wrong_shape, state.path()), std::runtime_error);

    const TemporaryFile bad_magic("optimizer-bad-magic.bin");
    bad_magic.text("not optimizer state");
    EXPECT_THROW(clnn::optim::load_state_dict(sgd, bad_magic.path()), std::runtime_error);
    const TemporaryFile truncated("optimizer-truncated.bin");
    truncated.text("CLNNOPT2");
    EXPECT_THROW(clnn::optim::load_state_dict(sgd, truncated.path()), std::runtime_error);
    EXPECT_THROW(
        clnn::optim::load_state_dict(sgd, TemporaryFile("missing-optimizer-state.bin").path()),
        std::runtime_error);
}

} // namespace
