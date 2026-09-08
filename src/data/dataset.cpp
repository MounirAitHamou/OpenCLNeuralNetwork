#include "clnn/data/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace clnn::data {
namespace {

std::vector<std::string> parse_csv_row(const std::string& line, const char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const auto character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == delimiter && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(character);
        }
    }
    if (quoted)
        throw std::runtime_error("unterminated quoted CSV field");
    fields.push_back(std::move(field));
    return fields;
}

float parse_number(const std::string& text, const std::size_t line_number) {
    std::size_t consumed = 0;
    try {
        const auto value = std::stof(text, &consumed);
        while (consumed < text.size() && std::isspace(static_cast<unsigned char>(text[consumed])))
            ++consumed;
        if (consumed != text.size())
            throw std::invalid_argument("trailing data");
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid numerical value '" + text + "' on CSV line " +
                                 std::to_string(line_number));
    }
}

} // namespace

TensorDataset::TensorDataset(std::vector<float> inputs, Shape input_shape,
                             std::vector<float> targets, Shape target_shape,
                             std::vector<std::size_t> class_labels)
    : inputs_(std::move(inputs)), input_shape_(std::move(input_shape)),
      targets_(std::move(targets)), target_shape_(std::move(target_shape)),
      class_labels_(std::move(class_labels)) {
    const auto input_elements = numel(input_shape_);
    const auto target_elements = numel(target_shape_);
    if (inputs_.size() % input_elements != 0 || targets_.size() % target_elements != 0 ||
        inputs_.size() / input_elements != targets_.size() / target_elements) {
        throw std::invalid_argument("dataset storage does not match its sample shapes");
    }
    if (!class_labels_.empty() && class_labels_.size() != size()) {
        throw std::invalid_argument("dataset must have one class label per sample");
    }
}

std::size_t TensorDataset::size() const noexcept {
    return inputs_.empty() ? 0 : inputs_.size() / numel(input_shape_);
}
const Shape& TensorDataset::input_shape() const noexcept {
    return input_shape_;
}
const Shape& TensorDataset::target_shape() const noexcept {
    return target_shape_;
}
bool TensorDataset::has_class_labels() const noexcept {
    return !class_labels_.empty();
}

TensorDataset TensorDataset::subset(const std::vector<std::size_t>& indices) const {
    const auto input_elements = numel(input_shape_);
    const auto target_elements = numel(target_shape_);
    std::vector<float> inputs(indices.size() * input_elements);
    std::vector<float> targets(indices.size() * target_elements);
    std::vector<std::size_t> labels;
    if (has_class_labels())
        labels.resize(indices.size());
    for (std::size_t destination = 0; destination < indices.size(); ++destination) {
        const auto source = indices[destination];
        if (source >= size())
            throw std::out_of_range("dataset subset index is out of range");
        std::copy_n(inputs_.begin() + static_cast<std::ptrdiff_t>(source * input_elements),
                    input_elements,
                    inputs.begin() + static_cast<std::ptrdiff_t>(destination * input_elements));
        std::copy_n(targets_.begin() + static_cast<std::ptrdiff_t>(source * target_elements),
                    target_elements,
                    targets.begin() + static_cast<std::ptrdiff_t>(destination * target_elements));
        if (!labels.empty())
            labels[destination] = class_labels_[source];
    }
    return TensorDataset(std::move(inputs), input_shape_, std::move(targets), target_shape_,
                         std::move(labels));
}

DatasetSplit split(const TensorDataset& dataset, const float training_fraction,
                   const float validation_fraction, const std::uint32_t seed) {
    if (training_fraction < 0.0F || validation_fraction < 0.0F ||
        training_fraction + validation_fraction > 1.0F) {
        throw std::invalid_argument(
            "dataset split fractions must be non-negative and sum to at most one");
    }
    std::vector<std::size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    const auto training_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(dataset.size()) * training_fraction));
    const auto validation_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(dataset.size()) * validation_fraction));
    const auto training_end = indices.begin() + static_cast<std::ptrdiff_t>(training_count);
    const auto validation_end = training_end + static_cast<std::ptrdiff_t>(validation_count);
    return {dataset.subset(std::vector<std::size_t>(indices.begin(), training_end)),
            dataset.subset(std::vector<std::size_t>(training_end, validation_end)),
            dataset.subset(std::vector<std::size_t>(validation_end, indices.end()))};
}

DataLoader::DataLoader(const TensorDataset& dataset, const std::size_t batch_size,
                       const Device device, const bool shuffle, const std::uint32_t seed,
                       const bool drop_last, const bool prefetch)
    : dataset_(&dataset), batch_size_(batch_size), device_(device), shuffle_(shuffle),
      drop_last_(drop_last), prefetch_(prefetch), generator_(seed), order_(dataset.size()) {
    if (batch_size == 0)
        throw std::invalid_argument("data loader batch size must be positive");
    std::iota(order_.begin(), order_.end(), 0);
    reset();
}

void DataLoader::reset(const std::optional<std::uint32_t> seed) {
    if (prefetched_.valid())
        static_cast<void>(prefetched_.get());
    if (seed.has_value())
        generator_.seed(*seed);
    cursor_ = 0;
    std::iota(order_.begin(), order_.end(), 0);
    if (shuffle_)
        std::shuffle(order_.begin(), order_.end(), generator_);
    schedule_prefetch();
}

std::optional<Batch> DataLoader::next() {
    if (!prefetch_)
        return next_synchronous();
    if (!prefetched_.valid())
        return std::nullopt;
    auto result = prefetched_.get();
    schedule_prefetch();
    return result;
}

std::optional<Batch> DataLoader::next_synchronous() {
    if (cursor_ >= order_.size())
        return std::nullopt;
    const auto remaining = order_.size() - cursor_;
    if (drop_last_ && remaining < batch_size_) {
        cursor_ = order_.size();
        return std::nullopt;
    }
    const auto count = std::min(batch_size_, remaining);
    const auto input_elements = numel(dataset_->input_shape_);
    const auto target_elements = numel(dataset_->target_shape_);
    std::vector<float> inputs(count * input_elements);
    std::vector<float> targets(count * target_elements);
    std::vector<std::size_t> labels;
    if (dataset_->has_class_labels())
        labels.resize(count);
    for (std::size_t row = 0; row < count; ++row) {
        const auto sample = order_[cursor_ + row];
        std::copy_n(
            dataset_->inputs_.begin() + static_cast<std::ptrdiff_t>(sample * input_elements),
            input_elements, inputs.begin() + static_cast<std::ptrdiff_t>(row * input_elements));
        std::copy_n(
            dataset_->targets_.begin() + static_cast<std::ptrdiff_t>(sample * target_elements),
            target_elements, targets.begin() + static_cast<std::ptrdiff_t>(row * target_elements));
        if (!labels.empty())
            labels[row] = dataset_->class_labels_[sample];
    }
    cursor_ += count;
    auto input_shape = dataset_->input_shape_;
    input_shape.insert(input_shape.begin(), count);
    auto target_shape = dataset_->target_shape_;
    target_shape.insert(target_shape.begin(), count);
    return Batch{Tensor(std::move(inputs), std::move(input_shape), false, {}, device_),
                 Tensor(std::move(targets), std::move(target_shape), false, {}, device_),
                 std::move(labels), count};
}

void DataLoader::schedule_prefetch() {
    if (!prefetch_ || cursor_ >= order_.size() ||
        (drop_last_ && order_.size() - cursor_ < batch_size_))
        return;
    prefetched_ = std::async(std::launch::async, [this] { return next_synchronous(); });
}

std::size_t DataLoader::batch_count() const noexcept {
    if (drop_last_)
        return dataset_->size() / batch_size_;
    return (dataset_->size() + batch_size_ - 1) / batch_size_;
}

bool DataLoader::prefetch_enabled() const noexcept {
    return prefetch_;
}

TensorDataset load_csv_numerical(const std::filesystem::path& path,
                                 const std::vector<std::string>& input_columns,
                                 const std::vector<std::string>& target_columns,
                                 const char delimiter) {
    if (input_columns.empty() || target_columns.empty()) {
        throw std::invalid_argument("CSV input and target column lists must not be empty");
    }
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open CSV dataset: " + path.string());
    std::string line;
    if (!std::getline(stream, line))
        throw std::runtime_error("CSV dataset is empty");
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    const auto header = parse_csv_row(line, delimiter);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (!columns.emplace(header[index], index).second) {
            throw std::runtime_error("CSV contains duplicate column '" + header[index] + "'");
        }
    }
    auto resolve = [&](const std::vector<std::string>& names) {
        std::vector<std::size_t> result;
        for (const auto& name : names) {
            const auto found = columns.find(name);
            if (found == columns.end())
                throw std::runtime_error("CSV column not found: " + name);
            result.push_back(found->second);
        }
        return result;
    };
    const auto input_indices = resolve(input_columns);
    const auto target_indices = resolve(target_columns);
    std::vector<float> inputs;
    std::vector<float> targets;
    std::size_t line_number = 1;
    while (std::getline(stream, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const auto fields = parse_csv_row(line, delimiter);
        if (fields.size() != header.size()) {
            throw std::runtime_error("CSV field count mismatch on line " +
                                     std::to_string(line_number));
        }
        for (const auto index : input_indices)
            inputs.push_back(parse_number(fields[index], line_number));
        for (const auto index : target_indices)
            targets.push_back(parse_number(fields[index], line_number));
    }
    return TensorDataset(std::move(inputs), {input_columns.size()}, std::move(targets),
                         {target_columns.size()});
}

TensorDataset load_cifar10_batch(const std::filesystem::path& path, const bool normalize) {
    constexpr std::size_t image_elements = 3 * 32 * 32;
    constexpr std::size_t record_size = image_elements + 1;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("cannot open CIFAR-10 batch: " + path.string());
    const auto byte_count = static_cast<std::size_t>(stream.tellg());
    if (byte_count == 0 || byte_count % record_size != 0) {
        throw std::runtime_error("invalid CIFAR-10 batch size");
    }
    stream.seekg(0);
    const auto samples = byte_count / record_size;
    std::vector<float> inputs(samples * image_elements);
    std::vector<float> targets(samples * 10, 0.0F);
    std::vector<std::size_t> labels(samples);
    std::vector<unsigned char> pixels(image_elements);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        unsigned char label = 0;
        stream.read(reinterpret_cast<char*>(&label), 1);
        stream.read(reinterpret_cast<char*>(pixels.data()),
                    static_cast<std::streamsize>(pixels.size()));
        if (!stream || label >= 10)
            throw std::runtime_error("corrupt CIFAR-10 record");
        labels[sample] = label;
        targets[sample * 10 + label] = 1.0F;
        for (std::size_t index = 0; index < pixels.size(); ++index) {
            inputs[sample * image_elements + index] =
                normalize ? static_cast<float>(pixels[index]) / 255.0F
                          : static_cast<float>(pixels[index]);
        }
    }
    return TensorDataset(std::move(inputs), {3, 32, 32}, std::move(targets), {10},
                         std::move(labels));
}

} // namespace clnn::data
