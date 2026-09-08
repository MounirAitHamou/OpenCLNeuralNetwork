#include "clnn/inspect.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace clnn {

TensorStatistics statistics(const Tensor& tensor) {
    const auto& values = tensor.data();
    if (values.empty())
        throw std::invalid_argument("cannot inspect an empty tensor");
    TensorStatistics result{std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            0,
                            0,
                            0,
                            0};
    double sum = 0.0;
    double squared_sum = 0.0;
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            ++result.non_finite;
            continue;
        }
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
        sum += value;
        squared_sum += static_cast<double>(value) * value;
    }
    const auto finite_count = values.size() - result.non_finite;
    if (finite_count == 0) {
        result.minimum = result.maximum = result.mean = result.standard_deviation = result.l2_norm =
            std::numeric_limits<float>::quiet_NaN();
        return result;
    }
    result.mean = static_cast<float>(sum / static_cast<double>(finite_count));
    const auto variance = squared_sum / static_cast<double>(finite_count) -
                          static_cast<double>(result.mean) * result.mean;
    result.standard_deviation = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    result.l2_norm = static_cast<float>(std::sqrt(squared_sum));
    return result;
}

std::string inspect(const Tensor& tensor, const std::size_t maximum_values) {
    const auto stats = statistics(tensor);
    std::ostringstream stream;
    stream << "Tensor(shape=" << shape_string(tensor.shape())
           << ", device=" << tensor.device().name() << ", requires_grad=" << std::boolalpha
           << tensor.requires_grad() << ")\n"
           << "  min=" << stats.minimum << " max=" << stats.maximum << " mean=" << stats.mean
           << " std=" << stats.standard_deviation << " l2=" << stats.l2_norm
           << " non_finite=" << stats.non_finite << "\n  values=[";
    const auto count = std::min(maximum_values, tensor.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0)
            stream << ", ";
        stream << tensor.data()[index];
    }
    if (count < tensor.size())
        stream << ", ...";
    stream << ']';
    return stream.str();
}

} // namespace clnn
