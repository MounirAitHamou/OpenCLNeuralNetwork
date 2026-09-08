#pragma once

#include "clnn/autograd/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace clnn::data {

class TensorDataset final {
  public:
    TensorDataset(std::vector<float> inputs, Shape input_shape, std::vector<float> targets,
                  Shape target_shape, std::vector<std::size_t> class_labels = {});

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const Shape& input_shape() const noexcept;
    [[nodiscard]] const Shape& target_shape() const noexcept;
    [[nodiscard]] bool has_class_labels() const noexcept;
    [[nodiscard]] TensorDataset subset(const std::vector<std::size_t>& indices) const;

  private:
    std::vector<float> inputs_;
    Shape input_shape_;
    std::vector<float> targets_;
    Shape target_shape_;
    std::vector<std::size_t> class_labels_;

    friend class DataLoader;
};

struct DatasetSplit final {
    TensorDataset training;
    TensorDataset validation;
    TensorDataset test;
};

[[nodiscard]] DatasetSplit split(const TensorDataset& dataset, float training_fraction,
                                 float validation_fraction, std::uint32_t seed);

struct Batch final {
    Tensor inputs;
    Tensor targets;
    std::vector<std::size_t> class_labels;
    std::size_t size = 0;
};

class DataLoader final {
  public:
    DataLoader(const TensorDataset& dataset, std::size_t batch_size, Device device = Device::cpu(),
               bool shuffle = false, std::uint32_t seed = 0, bool drop_last = false,
               bool prefetch = false);
    void reset(std::optional<std::uint32_t> seed = std::nullopt);
    [[nodiscard]] std::optional<Batch> next();
    [[nodiscard]] std::size_t batch_count() const noexcept;
    [[nodiscard]] bool prefetch_enabled() const noexcept;

  private:
    [[nodiscard]] std::optional<Batch> next_synchronous();
    void schedule_prefetch();
    const TensorDataset* dataset_;
    std::size_t batch_size_;
    Device device_;
    bool shuffle_;
    bool drop_last_;
    bool prefetch_;
    std::mt19937 generator_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ = 0;
    std::future<std::optional<Batch>> prefetched_;
};

[[nodiscard]] TensorDataset load_csv_numerical(const std::filesystem::path& path,
                                               const std::vector<std::string>& input_columns,
                                               const std::vector<std::string>& target_columns,
                                               char delimiter = ',');

[[nodiscard]] TensorDataset load_cifar10_batch(const std::filesystem::path& path,
                                               bool normalize = true);

} // namespace clnn::data
