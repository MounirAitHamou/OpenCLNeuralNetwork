#pragma once

#include <vector>
#include <numeric>

struct Dimensions {

    std::vector<size_t> dims;

    Dimensions() = default;

    Dimensions(std::initializer_list<size_t> d) : dims(d) {}

    Dimensions(const std::vector<size_t>& d) : dims(d) {}

    size_t getTotalElements() const;

    bool operator==(const Dimensions& other) const {
        return dims == other.dims;
    }

    bool operator!=(const Dimensions& other) const {
        return !(*this == other);
    }
};