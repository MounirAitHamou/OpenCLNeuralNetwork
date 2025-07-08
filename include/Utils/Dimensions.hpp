#pragma once

#include <vector>
#include <numeric>

/**
 * @struct Dimensions
 * @brief A utility structure to represent the dimensions of multi-dimensional data
 * (e.g., input/output shapes of layers, image dimensions).
 *
 * This structure allows for flexible representation of shapes, whether it's a
 * 1D vector, 2D matrix, or higher-dimensional tensor. It provides constructors
 * for easy initialization and a utility method to calculate the total number of elements.
 */
struct Dimensions {
    /**
     * @brief A `std::vector` storing the size of each dimension.
     * For example, for a 28x28 image, `dims` would be `{28, 28}`.
     * For a batch of 10 images of size 28x28, `dims` might be `{10, 28, 28}`.
     */
    std::vector<size_t> dims;

    // --- Constructors ---

    /**
     * @brief Default constructor.
     * Initializes `dims` as an empty vector, representing zero dimensions.
     */
    Dimensions() = default;

    /**
     * @brief Constructor that initializes dimensions using an initializer list.
     *
     * @param d An `std::initializer_list<size_t>` containing the sizes of each dimension.
     * Example: `Dimensions({28, 28})` for a 2D dimension.
     */
    Dimensions(std::initializer_list<size_t> d) : dims(d) {}

    /**
     * @brief Constructor that initializes dimensions from an existing `std::vector<size_t>`.
     *
     * @param d A `const std::vector<size_t>&` containing the sizes of each dimension.
     */
    Dimensions(const std::vector<size_t>& d) : dims(d) {}

    // --- Member Function ---

    /**
     * @brief Calculates the total number of elements represented by these dimensions.
     *
     * This is computed by multiplying all the dimension sizes together.
     * For example, for `Dimensions({2, 3, 4})`, it returns `2 * 3 * 4 = 24`.
     * If `dims` is empty, it returns 0.
     *
     * @return The total number of elements.
     */
    size_t getTotalElements() const;

    // --- Operators ---

    /**
     * @brief Overloads the equality operator (==) for `Dimensions` objects.
     *
     * Two `Dimensions` objects are considered equal if their underlying `dims`
     * vectors are identical (same size and same elements in the same order).
     *
     * @param other The `Dimensions` object to compare with.
     * @return `true` if the dimensions are equal, `false` otherwise.
     */
    bool operator==(const Dimensions& other) const {
        return dims == other.dims;
    }

    /**
     * @brief Overloads the inequality operator (!=) for `Dimensions` objects.
     *
     * @param other The `Dimensions` object to compare with.
     * @return `true` if the dimensions are not equal, `false` otherwise.
     */
    bool operator!=(const Dimensions& other) const {
        return !(*this == other);
    }
};