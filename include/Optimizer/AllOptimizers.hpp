#pragma once

#include "Optimizer/SGD/SGDOptimizer.hpp"
#include "Optimizer/Adam/AdamOptimizer.hpp"
#include "Optimizer/AdamW/AdamWOptimizer.hpp"

// This header file serves as a central point to include all defined optimizer types.
// By including this single header, other parts of the neural network framework
// (e.g., a Network class or an Optimizer factory) can easily access the
// declarations of all available optimizers without needing to include each one individually.
// This promotes modularity and simplifies dependency management.

// No classes or functions are defined directly in this header; it is purely for aggregation.