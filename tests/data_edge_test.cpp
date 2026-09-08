#include "clnn/clnn.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
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
        ASSERT_TRUE(stream);
    }

    void bytes(const std::vector<unsigned char>& contents) const {
        std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream);
        stream.write(reinterpret_cast<const char*>(contents.data()),
                     static_cast<std::streamsize>(contents.size()));
        ASSERT_TRUE(stream);
    }

  private:
    std::filesystem::path path_;
};

TEST(DataEdge, DatasetValidatesStorageLabelsAndSubsets) {
    EXPECT_THROW((clnn::data::TensorDataset({1, 2, 3}, {2}, {1, 2}, {1})), std::invalid_argument);
    EXPECT_THROW((clnn::data::TensorDataset({1, 2, 3, 4}, {2}, {1, 2}, {1}, {0})),
                 std::invalid_argument);

    const clnn::data::TensorDataset empty({}, {1}, {}, {1});
    EXPECT_EQ(empty.size(), 0U);
    EXPECT_FALSE(empty.has_class_labels());

    const clnn::data::TensorDataset dataset({1, 2, 3, 4}, {2}, {10, 20}, {1}, {7, 8});
    const auto subset = dataset.subset({1, 0});
    EXPECT_EQ(subset.size(), 2U);
    EXPECT_TRUE(subset.has_class_labels());
    clnn::data::DataLoader loader(subset, 2);
    const auto batch = loader.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->inputs.data(), (std::vector<float>{3, 4, 1, 2}));
    EXPECT_EQ(batch->targets.data(), (std::vector<float>{20, 10}));
    EXPECT_EQ(batch->class_labels, (std::vector<std::size_t>{8, 7}));
    EXPECT_THROW(static_cast<void>(dataset.subset({2})), std::out_of_range);
}

TEST(DataEdge, SplitAndLoaderValidateAndHandleDropLast) {
    const clnn::data::TensorDataset dataset({0, 1, 2, 3, 4}, {1}, {5, 6, 7, 8, 9}, {1});
    EXPECT_THROW(static_cast<void>(clnn::data::split(dataset, -0.1F, 0.2F, 1)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::data::split(dataset, 0.8F, 0.3F, 1)),
                 std::invalid_argument);
    EXPECT_THROW((clnn::data::DataLoader(dataset, 0)), std::invalid_argument);

    clnn::data::DataLoader dropped(dataset, 2, clnn::Device::cpu(), false, 0, true);
    EXPECT_EQ(dropped.batch_count(), 2U);
    ASSERT_TRUE(dropped.next().has_value());
    ASSERT_TRUE(dropped.next().has_value());
    EXPECT_FALSE(dropped.next().has_value());
    EXPECT_FALSE(dropped.next().has_value());

    clnn::data::DataLoader shuffled(dataset, 5, clnn::Device::cpu(), true, 17);
    shuffled.reset(91);
    const auto first = shuffled.next()->inputs.data();
    shuffled.reset(91);
    EXPECT_EQ(shuffled.next()->inputs.data(), first);

    const auto partitions = clnn::data::split(dataset, 0.0F, 0.0F, 3);
    EXPECT_EQ(partitions.training.size(), 0U);
    EXPECT_EQ(partitions.validation.size(), 0U);
    EXPECT_EQ(partitions.test.size(), dataset.size());
}

TEST(DataEdge, CsvSupportsQuotesWhitespaceBlankLinesAndCarriageReturns) {
    const TemporaryFile file("quoted.csv");
    file.text("\"feat\"\"ure\",target\r\n 1.5 ,2\r\n\r\n3,4\r\n");
    const auto dataset = clnn::data::load_csv_numerical(file.path(), {"feat\"ure"}, {"target"});
    EXPECT_EQ(dataset.size(), 2U);
    clnn::data::DataLoader loader(dataset, 2);
    const auto batch = loader.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->inputs.data(), (std::vector<float>{1.5F, 3.0F}));
    EXPECT_EQ(batch->targets.data(), (std::vector<float>{2.0F, 4.0F}));
}

TEST(DataEdge, CsvRejectsEveryMalformedInputClass) {
    const TemporaryFile missing("missing.csv");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(missing.path(), {"x"}, {"y"})),
                 std::runtime_error);

    const TemporaryFile valid("valid.csv");
    valid.text("x,y\n1,2\n");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(valid.path(), {}, {"y"})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(valid.path(), {"x"}, {})),
                 std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(clnn::data::load_csv_numerical(valid.path(), {"missing"}, {"y"})),
        std::runtime_error);

    const TemporaryFile empty("empty.csv");
    empty.text("");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(empty.path(), {"x"}, {"y"})),
                 std::runtime_error);

    const TemporaryFile duplicate("duplicate.csv");
    duplicate.text("x,x\n1,2\n");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(duplicate.path(), {"x"}, {"x"})),
                 std::runtime_error);

    const TemporaryFile quotes("quotes.csv");
    quotes.text("\"x,y\n1,2\n");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(quotes.path(), {"x"}, {"y"})),
                 std::runtime_error);

    const TemporaryFile fields("fields.csv");
    fields.text("x,y\n1,2,3\n");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(fields.path(), {"x"}, {"y"})),
                 std::runtime_error);

    const TemporaryFile number("number.csv");
    number.text("x,y\n1oops,2\n");
    EXPECT_THROW(static_cast<void>(clnn::data::load_csv_numerical(number.path(), {"x"}, {"y"})),
                 std::runtime_error);
}

TEST(DataEdge, CifarLoaderHandlesNormalizationLabelsAndCorruption) {
    constexpr std::size_t image_size = 3 * 32 * 32;
    std::vector<unsigned char> record(image_size + 1, 0);
    record[0] = 4;
    record[1] = 255;
    record[2] = 128;
    const TemporaryFile valid("cifar.bin");
    valid.bytes(record);

    const auto normalized = clnn::data::load_cifar10_batch(valid.path(), true);
    EXPECT_EQ(normalized.size(), 1U);
    EXPECT_EQ(normalized.input_shape(), (clnn::Shape{3, 32, 32}));
    EXPECT_EQ(normalized.target_shape(), (clnn::Shape{10}));
    EXPECT_TRUE(normalized.has_class_labels());
    auto normalized_batch = clnn::data::DataLoader(normalized, 1).next();
    ASSERT_TRUE(normalized_batch.has_value());
    EXPECT_FLOAT_EQ(normalized_batch->inputs.data()[0], 1.0F);
    EXPECT_NEAR(normalized_batch->inputs.data()[1], 128.0F / 255.0F, 1.0e-6F);
    EXPECT_EQ(normalized_batch->class_labels, (std::vector<std::size_t>{4}));
    EXPECT_FLOAT_EQ(normalized_batch->targets.data()[4], 1.0F);

    const auto raw = clnn::data::load_cifar10_batch(valid.path(), false);
    auto raw_batch = clnn::data::DataLoader(raw, 1).next();
    ASSERT_TRUE(raw_batch.has_value());
    EXPECT_FLOAT_EQ(raw_batch->inputs.data()[0], 255.0F);
    EXPECT_FLOAT_EQ(raw_batch->inputs.data()[1], 128.0F);

    const TemporaryFile missing("missing-cifar.bin");
    EXPECT_THROW(static_cast<void>(clnn::data::load_cifar10_batch(missing.path())),
                 std::runtime_error);
    const TemporaryFile empty("empty-cifar.bin");
    empty.bytes({});
    EXPECT_THROW(static_cast<void>(clnn::data::load_cifar10_batch(empty.path())),
                 std::runtime_error);
    const TemporaryFile wrong_size("wrong-size-cifar.bin");
    wrong_size.bytes({1, 2, 3});
    EXPECT_THROW(static_cast<void>(clnn::data::load_cifar10_batch(wrong_size.path())),
                 std::runtime_error);
    record[0] = 10;
    const TemporaryFile bad_label("bad-label-cifar.bin");
    bad_label.bytes(record);
    EXPECT_THROW(static_cast<void>(clnn::data::load_cifar10_batch(bad_label.path())),
                 std::runtime_error);
}

TEST(DataEdge, LoaderCanPlaceBatchesOnGpu) {
    if (!clnn::opencl_available())
        GTEST_SKIP() << "No OpenCL GPU is installed";
    const clnn::data::TensorDataset dataset({1, 2}, {1}, {3, 4}, {1});
    clnn::data::DataLoader loader(dataset, 2, clnn::Device::opencl());
    const auto batch = loader.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->inputs.device().type(), clnn::DeviceType::opencl);
    EXPECT_EQ(batch->targets.device().type(), clnn::DeviceType::opencl);
    EXPECT_EQ(batch->inputs.data(), (std::vector<float>{1, 2}));
}

TEST(DataEdge, PrefetchMatchesSynchronousIterationAndReset) {
    const clnn::data::TensorDataset dataset({0, 1, 2, 3, 4, 5}, {1}, {10, 11, 12, 13, 14, 15}, {1});
    clnn::data::DataLoader synchronous(dataset, 2, clnn::Device::cpu(), true, 37);
    clnn::data::DataLoader prefetched(dataset, 2, clnn::Device::cpu(), true, 37, false, true);
    EXPECT_TRUE(prefetched.prefetch_enabled());
    for (std::size_t batch_index = 0; batch_index < synchronous.batch_count(); ++batch_index) {
        const auto expected = synchronous.next();
        const auto actual = prefetched.next();
        ASSERT_TRUE(expected.has_value());
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(actual->inputs.data(), expected->inputs.data());
        EXPECT_EQ(actual->targets.data(), expected->targets.data());
    }
    EXPECT_FALSE(prefetched.next().has_value());

    prefetched.reset(91);
    const auto first = prefetched.next();
    ASSERT_TRUE(first.has_value());
    prefetched.reset(91);
    const auto repeated = prefetched.next();
    ASSERT_TRUE(repeated.has_value());
    EXPECT_EQ(repeated->inputs.data(), first->inputs.data());
}

} // namespace
